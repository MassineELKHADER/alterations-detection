from dataclasses import dataclass
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.special import gammaln, logsumexp
from skimage.draw import disk


@dataclass(frozen=True)
class CandidatePointCloud:
    """ Represents a point cloud of candidate alteration points """
    coordinates: np.ndarray
    transformed: np.ndarray
    quantized: np.ndarray
    original_values: np.ndarray
    image_shape: tuple[int, int]
    z_levels: int


@dataclass(frozen=True)
class ClusterCandidate:
    """ Represents a cluster of candidate alteration points """
    node_id: int
    member_indices: np.ndarray
    delta: float
    parent_delta: float
    lower_volume: float 
    upper_volume: float
    significance: float


def apply_gray_transform(values: np.ndarray, tau: float, kind: str) -> np.ndarray:
    """ Applies a gray-level transformation to the input values based on the specified kind and threshold tau."""
    values = values.astype(np.float32)
    transformed = np.full_like(values, np.inf, dtype=np.float32)
    valid = values >= tau
    if kind != "tanh":
        raise ValueError(f"Unsupported transform: {kind}")
    transformed[valid] = 1.0 + np.tanh(tau - values[valid])
    return transformed


def make_point_cloud(
    diff_map: np.ndarray,
    tau: float,
    transform: str,
    z_levels: int,
    max_points: int | None = None,
) -> CandidatePointCloud:
    """ Generates a point cloud of candidate alteration points from the input difference map"""
    transformed = apply_gray_transform(diff_map, tau=tau, kind=transform)
    finite_mask = np.isfinite(transformed)
    ys, xs = np.nonzero(finite_mask)
    finite_values = transformed[finite_mask]
    raw_values = diff_map[finite_mask]
    if max_points is not None and finite_values.size > max_points:
        keep = np.argpartition(raw_values, -max_points)[-max_points:]
        ys = ys[keep]
        xs = xs[keep]
        finite_values = finite_values[keep]
        raw_values = raw_values[keep]
    if finite_values.size == 0:
        return CandidatePointCloud(
            coordinates=np.zeros((0, 2), dtype=np.int32),
            transformed=np.zeros((0,), dtype=np.float32),
            quantized=np.zeros((0,), dtype=np.int32),
            original_values=np.zeros((0,), dtype=np.float32),
            image_shape=diff_map.shape,
            z_levels=z_levels,
        )

    z_min = float(finite_values.min())
    z_max = float(finite_values.max())
    if abs(z_max - z_min) < 1e-9:
        quantized = np.zeros_like(finite_values, dtype=np.int32)
    else:
        scaled = (finite_values - z_min) / (z_max - z_min)
        quantized = np.clip(np.round(scaled * (z_levels - 1)), 0, z_levels - 1).astype(np.int32)
    coordinates = np.stack([ys, xs], axis=1).astype(np.int32)
    return CandidatePointCloud(
        coordinates=coordinates,
        transformed=finite_values.astype(np.float32),
        quantized=quantized,
        original_values=raw_values.astype(np.float32),
        image_shape=diff_map.shape,
        z_levels=z_levels,
    )


def pairwise_distance_matrix(point_cloud: CandidatePointCloud, c_weight: float) -> np.ndarray:
    """ Computes a pairwise distance matrix for the candidate points in the point cloud, combining spatial and intensity differences."""
    coords = point_cloud.coordinates.astype(np.float32)
    z2 = np.square(point_cloud.quantized.astype(np.float32))
    delta = coords[:, None, :] - coords[None, :, :]
    spatial = np.sqrt(np.sum(delta * delta, axis=-1))
    distance = np.sqrt(spatial * spatial + c_weight * (z2[:, None] + z2[None, :]))
    np.fill_diagonal(distance, 0.0)
    return distance


def _condensed(square_matrix) -> np.ndarray:
    """ Converts a square distance matrix to condensed form as required by scipy's hierarchical clustering functions."""
    rows, cols = np.triu_indices(square_matrix.shape[0], 1)
    return square_matrix[rows, cols]


def _discrete_dilated_volume(
    cluster_xy: np.ndarray,
    cluster_z: np.ndarray,
    radius: float,
    image_shape: tuple[int, int],
    z_levels: int,
    c_weight: float,
) -> float:
    """
    Computes the volume of the union of discrete dilated spheres around the cluster points in the 
    3D space defined by spatial coordinates and quantized intensity levels.
    
    - it is needed to compute the NFA of a cluster candidate by estimating the 
    probability of observing a cluster of that size and density under a random model.
    """
    if cluster_xy.size == 0 or radius <= 0:
        return 0.0
    height, width = image_shape
    occupancy = np.zeros((z_levels, height, width), dtype=bool)
    z_grid = np.arange(z_levels, dtype=np.float32)
    for (y, x), zi in zip(cluster_xy, cluster_z, strict=True):
        valid_z = np.where(c_weight * (float(zi) ** 2 + z_grid ** 2) <= radius ** 2)[0]
        for z in valid_z:
            radius_sq = radius ** 2 - c_weight * (float(zi) ** 2 + float(z) ** 2)
            if radius_sq <= 0:
                continue
            rr, cc = disk((int(y), int(x)), float(np.sqrt(radius_sq)), shape=(height, width))
            occupancy[z, rr, cc] = True
    cube_volume = float(height * width * z_levels)
    return float(occupancy.sum()) / cube_volume


def _log_nfa(k: int, m: int, lower_volume: float, upper_volume: float) -> float:
    """ Computes the logarithm of the Number of False Alarms (NFA) for a cluster candidate based on its size (k), 
    total number of points (m) and the estimated volumes under the random model."""
    if k <= 0 or lower_volume <= 0.0 or upper_volume >= 1.0:
        return np.inf
    terms = np.arange(k, m + 1, dtype=np.int64)
    log_choose = gammaln(m + 1) - gammaln(terms + 1) - gammaln(m - terms + 1)
    log_inside = terms * np.log(max(lower_volume, 1e-12))
    log_outside = (m - terms) * np.log(max(1.0 - upper_volume, 1e-12))
    return float(logsumexp(log_choose + log_inside + log_outside))


def generate_cluster_candidates(point_cloud: CandidatePointCloud, c_weight: float) -> list[ClusterCandidate]:
    """ Generates a list of cluster candidates from the input point cloud by performing hierarchical 
    clustering and evaluating the significance of each cluster based on its size and density."""
    if len(point_cloud.coordinates) < 2:
        return []

    distance = pairwise_distance_matrix(point_cloud, c_weight=c_weight)
    hierarchy = linkage(_condensed(distance), method="single")
    n_leaves = len(point_cloud.coordinates)

    members: dict[int, np.ndarray] = {idx: np.array([idx], dtype=np.int32) for idx in range(n_leaves)}
    merge_distance = {idx: 0.0 for idx in range(n_leaves)}
    parent_distance: dict[int, float] = {}
    candidates: list[ClusterCandidate] = []

    for step, row in enumerate(hierarchy, start=0):
        left = int(row[0])
        right = int(row[1])
        node_id = n_leaves + step
        merged_members = np.concatenate([members[left], members[right]])
        members[node_id] = merged_members
        delta = float(row[2])
        merge_distance[node_id] = delta
        parent_distance[left] = delta
        parent_distance[right] = delta

    last_node = n_leaves + len(hierarchy) - 1
    parent_distance[last_node] = merge_distance[last_node]

    for node_id in range(n_leaves, n_leaves + len(hierarchy)):
        member_indices = members[node_id]
        cluster_xy = point_cloud.coordinates[member_indices]
        cluster_z = point_cloud.quantized[member_indices]
        delta = merge_distance[node_id]
        parent_delta_value = parent_distance[node_id]
        lower_volume = _discrete_dilated_volume(
            cluster_xy=cluster_xy,
            cluster_z=cluster_z,
            radius=delta / 2.0,
            image_shape=point_cloud.image_shape,
            z_levels=point_cloud.z_levels,
            c_weight=c_weight,
        )
        upper_volume = _discrete_dilated_volume(
            cluster_xy=cluster_xy,
            cluster_z=cluster_z,
            radius=parent_delta_value,
            image_shape=point_cloud.image_shape,
            z_levels=point_cloud.z_levels,
            c_weight=c_weight,
        )
        log_nfa = _log_nfa(
            k=len(member_indices),
            m=len(point_cloud.coordinates),
            lower_volume=lower_volume,
            upper_volume=upper_volume,
        )
        significance = -log_nfa if np.isfinite(log_nfa) else float("-inf")
        candidates.append(
            ClusterCandidate(
                node_id=node_id,
                member_indices=member_indices,
                delta=delta,
                parent_delta=parent_delta_value,
                lower_volume=lower_volume,
                upper_volume=upper_volume,
                significance=significance,
            )
        )
    return candidates

def maximal_meaningful_clusters(candidates: list[ClusterCandidate], top_k: int) -> list[ClusterCandidate]:
    """ Selects the top_k most meaningful clusters from the list of candidates while ensuring 
        that selected clusters do not share member points."""
    selected: list[ClusterCandidate] = []
    occupied: list[set[int]] = []
    for candidate in sorted(candidates, key=lambda item: item.significance, reverse=True):
        member_set = set(candidate.member_indices.tolist())
        if any(member_set & current for current in occupied):
            continue
        selected.append(candidate)
        occupied.append(member_set)
        if len(selected) >= top_k:
            break
    return selected
