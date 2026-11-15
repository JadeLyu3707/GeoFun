# @author: yl12783, Nov. 2026, New York

import os
from dataclasses import dataclass

import h5py as h5
import numpy as np

OBSTACLE = 1   # MeshGraphNets node type of the actuator


def tet_volumes(P: np.ndarray) -> np.ndarray:
    """P: (K, 4, 3) vertex coordinates -> (K,) volumes."""
    return np.abs(np.einsum("ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])) / 6


@dataclass
class SnapshotMetadata:

    dataset_name: str
    split_name: str
    traj: str
    t: int
    with_obstacle: bool
    n_points: int
    n_plate_points: int
    n_obstacle_points: int
    n_tets: int
    volume: float
    plate_volume: float
    obstacle_volume: float


class PlateSnapshot:
    """One time step of one deforming_plate trajectory.

    points : (n, 3)    the dataset's mesh nodes, used unchanged as the kernel centres x_i
    tets   : (K, 4, 3) vertex coordinates of the dataset's tetrahedra = ground-truth geometry
    """

    def __init__(
        self,
        base_path: str,
        dataset_name: str = "deforming_plate",
        split_name: str = "test",
        traj: str = "traj_00000",
        t: int = 0,
        with_obstacle: bool = True,
    ):
        self.data_path = os.path.join(base_path, dataset_name, "data", split_name, f"{dataset_name}_{split_name}.h5")

        # Copy params
        params = locals()
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):
                setattr(self, k, v)

        assert os.path.isfile(self.data_path), f"No HDF5 file at {self.data_path} (run scripts/tfrecord_to_h5.py)"
        with h5.File(self.data_path, "r") as f:
            g = f[traj]
            X = g["world_pos"][t].astype(np.float64)
            node_type = g["node_type"][:]
            cells = g["cells"][:]

        keep = np.ones(len(node_type), bool) if with_obstacle else node_type != OBSTACLE
        tet_idx = cells[keep[cells].all(1)]

        self.points = X[keep]
        self.node_type = node_type[keep]
        self.tets = X[tet_idx]
        self.is_obstacle_tet = (node_type[tet_idx] == OBSTACLE).all(1)
        vol = tet_volumes(self.tets)

        self.metadata = SnapshotMetadata(
            dataset_name=dataset_name,
            split_name=split_name,
            traj=traj,
            t=t,
            with_obstacle=with_obstacle,
            n_points=len(self.points),
            n_plate_points=int((self.node_type != OBSTACLE).sum()),
            n_obstacle_points=int((self.node_type == OBSTACLE).sum()),
            n_tets=len(self.tets),
            volume=float(vol.sum()),
            plate_volume=float(vol[~self.is_obstacle_tet].sum()),
            obstacle_volume=float(vol[self.is_obstacle_tet].sum()),
        )

    @property
    def obstacle_points(self) -> np.ndarray:
        return self.points[self.node_type == OBSTACLE]

    @property
    def plate_points(self) -> np.ndarray:
        return self.points[self.node_type != OBSTACLE]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.split_name}/{self.traj} t={self.t}, {len(self.points)} points>"


if __name__ == "__main__":
    snap = PlateSnapshot(base_path="/scratch/yl12783/datasets/meshgraphnets")
    print(snap)
    print(snap.metadata)
