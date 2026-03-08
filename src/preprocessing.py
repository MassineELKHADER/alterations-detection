from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.measure import label, regionprops
from skimage.transform import resize


@dataclass(frozen=True)
class RegistrationResult:
    """Result of aligning two images using feature-based registration."""
    aligned: np.ndarray
    matrix: np.ndarray
    inliers: int


@dataclass(frozen=True)
class Roi:
    """ Represents a region of interest (ROI) in an image, defined by its bounding box coordinates"""
    y0: int
    y1: int
    x0: int
    x1: int

    def as_slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

def _resize_for_registration(image: np.ndarray, max_side: int = 1200) -> tuple[np.ndarray, float]:
    """ Resizes the image to ensure its largest side does not exceed max_side, while maintaining aspect ratio.
                    # SampleViolin1 : (2102, 2910, 3)
                    # WoodSample1 : (315, 697, 3)
                    # WoodSample2 : (376, 646, 4)
        It returns the resized image and the scale factor used for resizing. If the image is already smaller than max_side, it returns the original image and a scale factor of 1.0.
    """
    height, width = image.shape[:2]
    current = max(height, width)
    if current <= max_side:
        return image, 1.0
    scale = max_side / current
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def register_to_reference(reference: np.ndarray, image: np.ndarray) -> RegistrationResult:
    """ 
        Aligns the input image to the reference image using feature-based registration. 
        It detects keypoints and descriptors in both images using SIFT, matches them using a brute-force matcher,
        and estimates a homography matrix to align the image to the reference. The function returns the aligned image, 
        the homography matrix and the number of inliers used in the estimation 
     """
    reference_small, ref_scale = _resize_for_registration(reference)
    image_small, img_scale = _resize_for_registration(image)
    gray_ref = cv2.cvtColor(reference_small, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.cvtColor(image_small, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp_ref, desc_ref = sift.detectAndCompute(gray_ref, None)
    kp_img, desc_img = sift.detectAndCompute(gray_img, None)
    if desc_ref is None or desc_img is None:
        identity = np.eye(3, dtype=np.float32)
        return RegistrationResult(aligned=image.copy(), matrix=identity, inliers=0)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc_img, desc_ref, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)

    if len(good) < 4:
        identity = np.eye(3, dtype=np.float32)
        return RegistrationResult(aligned=image.copy(), matrix=identity, inliers=0)

    src = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None:
        matrix = np.eye(3, dtype=np.float32)
        inliers = 0
    else:
        inliers = int(mask.sum()) if mask is not None else 0
    scale_ref = np.array([[ref_scale, 0.0, 0.0], [0.0, ref_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scale_img = np.array([[img_scale, 0.0, 0.0], [0.0, img_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    matrix = np.linalg.inv(scale_ref) @ matrix @ scale_img
    aligned = cv2.warpPerspective(image, matrix, (reference.shape[1], reference.shape[0]))
    return RegistrationResult(aligned=aligned, matrix=matrix, inliers=inliers)



def match_illumination(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    """ Adjusts the illumination of the input image to match that of the reference image using a simple scaling approach.
        It computes the median color of both images and scales the input image's colors to match the reference's median color. 
    """
    ref = reference.astype(np.float32)
    img = image.astype(np.float32)
    ref_median = np.median(ref.reshape(-1, 3), axis=0)
    img_median = np.median(img.reshape(-1, 3), axis=0)
    ratio = np.divide(ref_median, np.maximum(img_median, 1e-6))
    corrected = np.clip(img * ratio[None, None, :], 0, 255)
    return corrected.astype(np.uint8)


def color_difference_map(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    """ Computes a color difference map between the reference and input images using the CIEDE2000 metric """
    lab_ref = rgb2lab(reference)
    lab_img = rgb2lab(image)
    return deltaE_ciede2000(lab_ref, lab_img).astype(np.float32)


def auto_roi_from_difference(
    reference: np.ndarray,
    image: np.ndarray,
    percentile: float,
    padding: int,
    min_component_area: int,
) -> Roi:
    """ Automatically determines a region of interest (ROI) based on the color difference between the reference and input images. 
            - First applied a mask to identify areas of significant color difference (by thresholding)
            - Then, it identifies connected components in the binary mask and selects the largest component that meets a minimum area criterion.
            - Finally, it computes the bounding box of the selected component and applies padding.
    
    """
    diff = color_difference_map(reference, image)
    threshold = np.percentile(diff, percentile)
    binary = diff >= threshold
    components = label(binary)
    props = [prop for prop in regionprops(components) if prop.area >= min_component_area]
    if not props:
        return Roi(0, reference.shape[0], 0, reference.shape[1])
    largest = max(props, key=lambda prop: prop.area)
    y0, x0, y1, x1 = largest.bbox
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(reference.shape[0], y1 + padding)
    x1 = min(reference.shape[1], x1 + padding)
    return Roi(y0=y0, y1=y1, x0=x0, x1=x1)


def crop_roi(image, roi) -> np.ndarray:
    return image[roi.as_slices()]


def resize_max_side(image, max_side) -> np.ndarray:
    height, width = image.shape[:2]
    current = max(height, width)
    if current <= max_side:
        return image
    scale = max_side / current
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = resize(image, (new_height, new_width), order=1, preserve_range=True, anti_aliasing=True)
    return resized.astype(image.dtype)


def choose_best_shot(reference: np.ndarray, candidates: Iterable[np.ndarray]) -> tuple[int, RegistrationResult]:
    """ 
    Given a reference image and a list of candidate images, this function evaluates each candidate by 
    aligning it to the reference, adjusting its illumination and computing a color difference score.
    """
    best_index = -1
    best_registration: RegistrationResult | None = None
    best_score = None
    for idx, candidate in enumerate(candidates):
        registration = register_to_reference(reference, candidate)
        corrected = match_illumination(reference, registration.aligned)
        diff = color_difference_map(reference, corrected)
        score = float(np.median(diff)) - 0.01 * registration.inliers 
        # the score is better for corrected images that are more similar to the reference (lower median difference) 
        # and that have more inliers in the registration (more features matched)
        
        if best_score is None or score < best_score:
            best_score = score
            best_index = idx
            best_registration = RegistrationResult(corrected, registration.matrix, registration.inliers)
    if best_registration is None:
        raise ValueError("No candidate frames were provided.")
    return best_index, best_registration
