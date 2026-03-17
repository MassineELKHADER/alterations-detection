from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.cluster.hierarchy import linkage
from scipy.special import gammaln, logsumexp
from skimage.draw import disk
import time
from tqdm import tqdm


@dataclass(frozen=True)
class CandidatePointCloud:
    """Represents a point cloud of candidate alteration points."""
    coordinates: np.ndarray
    transformed: np.ndarray
    quantized: np.ndarray
    original_values: np.ndarray
    image_shape: tuple[int, int]
    z_levels: int


@dataclass(frozen=True)
class ClusterCandidate:
    """Represents a cluster of candidate alteration points."""
    node_id: int
    member_indices: np.ndarray
    delta: float
    parent_delta: float
    lower_volume: float
    upper_volume: float
    significance: float


def apply_gray_transform(values: np.ndarray, tau: float, kind: str) -> np.ndarray:
    """Applies a gray-level transformation to the input values based on the specified kind and threshold tau."""
    values = values.astype(np.float32)
    transformed = np.full_like(values, np.inf, dtype=np.float32)

    if kind == "tanh":
        valid = values >= tau
        transformed[valid] = 1.0 + np.tanh(tau - values[valid])
        return transformed

    if kind == "inverse":
        if tau == 0:
            raise ValueError("tau must be non-zero for the inverse transform")
        valid = values >= tau
        # transformed[valid] = 1.0 / (1.0 + (values[valid] / tau) ** 2)  # WRONG
        transformed[valid] = 1.0 / (values[valid] - tau + 1e-9)  # Paper: f1(x) = 1/(x-tau)
        return transformed

    raise ValueError(f"Unsupported transform: {kind}")


def make_point_cloud(
    diff_map: np.ndarray,
    tau: float,
    transform: str,
    z_levels: int,
    max_points: int | None = None,
) -> CandidatePointCloud:
    """Generates a point cloud of candidate alteration points from the input difference map."""
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
    """Computes a pairwise distance matrix for the candidate points in the point cloud."""
    coords = point_cloud.coordinates.astype(np.float32)

    if c_weight == 0.0:
        delta = coords[:, None, :] - coords[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        np.fill_diagonal(distance, 0.0)
        return distance

    # FIX: use float transformed z, not quantized integers (same fix as model.py)
    # z2 = np.square(point_cloud.quantized.astype(np.float32))  # WRONG
    z2 = np.square(point_cloud.transformed.astype(np.float32))  # FIXED
    delta = coords[:, None, :] - coords[None, :, :]
    spatial = np.sqrt(np.sum(delta * delta, axis=-1))
    distance = np.sqrt(spatial * spatial + c_weight * (z2[:, None] + z2[None, :]))
    np.fill_diagonal(distance, 0.0)
    return distance


def _condensed(square_matrix: np.ndarray) -> np.ndarray:
    """Converts a square distance matrix to condensed form."""
    rows, cols = np.triu_indices(square_matrix.shape[0], 1)
    return square_matrix[rows, cols]


_disk_offset_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def get_disk_offsets(radius: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Caches the same rounded disk offsets as model.py, so this backend stays
    behaviorally aligned with the current reference implementation.
    """
    key = round(radius, 4)
    if key not in _disk_offset_cache:
        r_int = int(np.ceil(radius))
        patch_shape = (2 * r_int + 1, 2 * r_int + 1)
        center = (r_int, r_int)
        rr, cc = disk(center, radius, shape=patch_shape)
        _disk_offset_cache[key] = (
            (rr - r_int).astype(np.int32),
            (cc - r_int).astype(np.int32),
        )
    return _disk_offset_cache[key]


@njit(cache=True)
def _paint_offsets(
    occupancy: np.ndarray,
    z_index: int,
    ys: np.ndarray,
    xs: np.ndarray,
    rr_offset: np.ndarray,
    cc_offset: np.ndarray,
) -> None:
    height = occupancy.shape[1]
    width = occupancy.shape[2]

    for point_idx in range(ys.shape[0]):
        base_y = ys[point_idx]
        base_x = xs[point_idx]

        for offset_idx in range(rr_offset.shape[0]):
            rr = base_y + rr_offset[offset_idx]
            cc = base_x + cc_offset[offset_idx]

            if 0 <= rr < height and 0 <= cc < width:
                occupancy[z_index, rr, cc] = True


def discrete_dilated_volume_numba(
    cluster_xy: np.ndarray,
    cluster_z: np.ndarray,
    radius: float,
    image_shape: tuple[int, int],
    z_levels: int,
    c_weight: float,
) -> float:
    """
    Computes the same discrete dilated volume as model.py, but uses a Numba
    kernel for the repeated occupancy writes.
    """
    if cluster_xy.size == 0 or radius <= 0:
        return 0.0

    height, width = image_shape

    if c_weight == 0.0:
        occupancy_2d = np.zeros((height, width), dtype=bool)
        rr_off, cc_off = get_disk_offsets(radius)
        for (y, x) in cluster_xy:
            rr = rr_off + int(y)
            cc = cc_off + int(x)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            occupancy_2d[rr[valid], cc[valid]] = True
        return float(occupancy_2d.sum()) / float(height * width)

    occupancy = np.zeros((z_levels, height, width), dtype=np.bool_)
    z_grid = np.arange(z_levels, dtype=np.float32)
    cluster_z = cluster_z.astype(np.int32, copy=False)

    for zi in np.unique(cluster_z):
        points = cluster_xy[cluster_z == zi]
        ys = points[:, 0].astype(np.int32, copy=False)
        xs = points[:, 1].astype(np.int32, copy=False)

        valid_z = np.where(c_weight * (float(zi) ** 2 + z_grid ** 2) <= radius ** 2)[0]
        for z in valid_z:
            radius_sq = radius ** 2 - c_weight * (float(zi) ** 2 + float(z) ** 2)
            if radius_sq <= 0:
                continue

            rr_offset, cc_offset = get_disk_offsets(float(np.sqrt(radius_sq)))
            _paint_offsets(occupancy, int(z), ys, xs, rr_offset, cc_offset)

    cube_volume = float(height * width * z_levels)
    return float(occupancy.sum()) / cube_volume


def _log_nfa(k: int, m: int, lower_volume: float, upper_volume: float) -> float:
    """Computes the logarithm of the Number of False Alarms for a cluster candidate."""
    if k <= 0 or lower_volume <= 0.0 or upper_volume >= 1.0:
        return np.inf

    terms = np.arange(k, m + 1, dtype=np.int64)
    log_choose = gammaln(m + 1) - gammaln(terms + 1) - gammaln(m - terms + 1)
    log_inside = terms * np.log(max(lower_volume, 1e-12))
    log_outside = (m - terms) * np.log(max(1.0 - upper_volume, 1e-12))
    return float(logsumexp(log_choose + log_inside + log_outside))


def generate_cluster_candidates(point_cloud: CandidatePointCloud, c_weight: float) -> list[ClusterCandidate]:
    """Generates cluster candidates using the Numba-accelerated volume computation."""
    if len(point_cloud.coordinates) < 2:
        return []

    timings = {
        "distance_matrix": 0.0,
        "linkage": 0.0,
        "hierarchy_processing": 0.0,
        "lower_volume": 0.0,
        "upper_volume": 0.0,
        "log_nfa": 0.0,
        "total": 0.0,
    }

    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    distance = pairwise_distance_matrix(point_cloud, c_weight=c_weight)
    timings["distance_matrix"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hierarchy = linkage(_condensed(distance), method="single")
    timings["linkage"] = time.perf_counter() - t0
    n_leaves = len(point_cloud.coordinates)

    members: dict[int, np.ndarray] = {idx: np.array([idx], dtype=np.int32) for idx in range(n_leaves)}
    merge_distance = {idx: 0.0 for idx in range(n_leaves)}
    parent_distance: dict[int, float] = {}
    candidates: list[ClusterCandidate] = []

    t0 = time.perf_counter()
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
    timings["hierarchy_processing"] = time.perf_counter() - t0

    total_nodes = len(hierarchy)
    M = len(point_cloud.coordinates)
    # Skip clusters larger than this fraction of M — upper_volume → 1 anyway (NFA = inf)
    large_cluster_skip = int(0.8 * M)

    for node_id in tqdm(
        range(n_leaves, n_leaves + total_nodes),
        total=total_nodes,
        desc="Evaluating clusters",
    ):
        member_indices = members[node_id]
        k = len(member_indices)
        delta = merge_distance[node_id]
        parent_delta_value = parent_distance[node_id]

        # Early exit 1: large clusters are trivially insignificant (upper_volume → 1)
        if k > large_cluster_skip:
            candidates.append(ClusterCandidate(
                node_id=node_id, member_indices=member_indices,
                delta=delta, parent_delta=parent_delta_value,
                lower_volume=0.0, upper_volume=1.0, significance=float("-inf"),
            ))
            continue

        cluster_xy = point_cloud.coordinates[member_indices]
        cluster_z = point_cloud.quantized[member_indices]

        # Early exit 2: zero merge distance → lower_volume will be 0 → NFA = inf
        if delta == 0.0:
            candidates.append(ClusterCandidate(
                node_id=node_id, member_indices=member_indices,
                delta=delta, parent_delta=parent_delta_value,
                lower_volume=0.0, upper_volume=0.0, significance=float("-inf"),
            ))
            continue

        t0 = time.perf_counter()
        lower_volume = discrete_dilated_volume_numba(
            cluster_xy=cluster_xy, cluster_z=cluster_z,
            radius=delta / 2.0, image_shape=point_cloud.image_shape,
            z_levels=point_cloud.z_levels, c_weight=c_weight,
        )
        timings["lower_volume"] += time.perf_counter() - t0

        # Early exit 3: lower_volume == 0 → NFA = inf regardless of upper_volume (saves expensive upper computation)
        if lower_volume <= 0.0:
            candidates.append(ClusterCandidate(
                node_id=node_id, member_indices=member_indices,
                delta=delta, parent_delta=parent_delta_value,
                lower_volume=0.0, upper_volume=0.0, significance=float("-inf"),
            ))
            continue

        t0 = time.perf_counter()
        upper_volume = discrete_dilated_volume_numba(
            cluster_xy=cluster_xy, cluster_z=cluster_z,
            radius=parent_delta_value, image_shape=point_cloud.image_shape,
            z_levels=point_cloud.z_levels, c_weight=c_weight,
        )
        timings["upper_volume"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        log_nfa = _log_nfa(k=k, m=M, lower_volume=lower_volume, upper_volume=upper_volume)
        timings["log_nfa"] += time.perf_counter() - t0
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

    timings["total"] = time.perf_counter() - t_total_start

    print("\n========== Timing report ==========")
    for key, value in timings.items():
        print(f"{key:20s}: {value:.4f} sec")
    print("===================================\n")

    return candidates


def maximal_meaningful_clusters(candidates: list[ClusterCandidate], top_k: int) -> list[ClusterCandidate]:
    """Selects the top_k most meaningful clusters from the list of candidates while ensuring no overlap."""
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


__all__ = [
    "CandidatePointCloud",
    "ClusterCandidate",
    "apply_gray_transform",
    "make_point_cloud",
    "pairwise_distance_matrix",
    "discrete_dilated_volume_numba",
    "generate_cluster_candidates",
    "maximal_meaningful_clusters",
]
