from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig
from .model import (
    CandidatePointCloud,
    ClusterCandidate,
    generate_cluster_candidates,
    make_point_cloud,
    maximal_meaningful_clusters,
)

# The DetectionResult class encapsulates the results of the alteration detection process, 
# including the difference map, point cloud, cluster candidates and selected clusters.
@dataclass(frozen=True)
class DetectionResult:
    diff_map: np.ndarray
    point_cloud: CandidatePointCloud
    candidates: list[ClusterCandidate]
    selected_clusters: list[ClusterCandidate]

    def cluster_mask(self, cluster: ClusterCandidate) -> np.ndarray:
        """ Generates a binary mask for the given cluster candidate by marking the pixels corresponding to the cluster points."""
        mask = np.zeros(self.diff_map.shape, dtype=bool)
        coords = self.point_cloud.coordinates[cluster.member_indices]
        mask[coords[:, 0], coords[:, 1]] = True
        return mask

# The AContrarioDetector class encapsulates the alteration detection process using an a-contrario approach.
class AContrarioDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def run(self, diff_map: np.ndarray) -> DetectionResult:
        """" Runs the alteration detection process on the input difference map by generating a point cloud of candidate alteration points,"""
        point_cloud = make_point_cloud(
            diff_map,
            tau=self.config.tau,
            transform=self.config.transform,
            z_levels=self.config.quantization_levels,
            max_points=self.config.max_points,
        )
        candidates = generate_cluster_candidates(point_cloud, c_weight=self.config.c_weight)
        selected = maximal_meaningful_clusters(candidates, top_k=self.config.significance_top_k)
        return DetectionResult(diff_map=diff_map, point_cloud=point_cloud, candidates=candidates, selected_clusters=selected)
