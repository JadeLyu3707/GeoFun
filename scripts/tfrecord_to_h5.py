# @author: yl12783, Nov. 2026, New York
#
# Convert the DeepMind MeshGraphNets tfrecords (e.g. deforming_plate) to HDF5, one group per trajectory.
#
# <out_root>/<dataset_name>/data/<split>/<dataset_name>_<split>.h5
#   attrs: dataset_name, n_trajectories, n_steps, collision_radius
#   traj_00000/
#       cells       (C, 4)    int32    static, tetrahedra
#       mesh_edges  (2, E)    int64    static, bidirectional, derived from cells
#       node_type   (N,)      int32    static (0 normal, 1 obstacle, 3 handle)
#       mesh_pos    (N, 3)    float32  static, reference configuration
#       world_pos   (T, N, 3) float32  dynamic
#       stress      (T, N, 1) float32  dynamic

import os
import json
import argparse
import numpy as np
import h5py as h5
from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")


def parse_record(proto, meta):
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"]).numpy()
        if field["type"] == "static":
            data = data[0]
        out[key] = data
    return out


def cells_to_edges(cells):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = np.concatenate([cells[:, [i, j]] for i, j in pairs], axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return np.stack([np.concatenate([edges[:, 0], edges[:, 1]]),
                     np.concatenate([edges[:, 1], edges[:, 0]])]).astype(np.int64)


def convert_split(in_dir, out_root, dataset_name, split, meta):
    out_dir = os.path.join(out_root, dataset_name, "data", split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_{split}.h5")
    tmp_path = out_path + ".tmp"
    records = tf.data.TFRecordDataset(os.path.join(in_dir, f"{split}.tfrecord"))
    n_traj = 0
    with h5.File(tmp_path, "w") as f:
        for i, proto in enumerate(tqdm(records, desc=split)):
            traj = parse_record(proto, meta)
            g = f.create_group(f"traj_{i:05d}")
            g.create_dataset("cells", data=traj["cells"].astype(np.int32))
            g.create_dataset("mesh_edges", data=cells_to_edges(traj["cells"]))
            g.create_dataset("node_type", data=traj["node_type"][:, 0].astype(np.int32))
            g.create_dataset("mesh_pos", data=traj["mesh_pos"].astype(np.float32))
            n_nodes = traj["world_pos"].shape[1]
            g.create_dataset("world_pos", data=traj["world_pos"].astype(np.float32), chunks=(2, n_nodes, 3))
            g.create_dataset("stress", data=traj["stress"].astype(np.float32), chunks=(2, n_nodes, 1))
            n_traj += 1
        f.attrs["dataset_name"] = dataset_name
        f.attrs["n_trajectories"] = n_traj
        f.attrs["n_steps"] = meta["trajectory_length"]
        f.attrs["collision_radius"] = meta.get("collision_radius", 0.0)
    os.replace(tmp_path, out_path)
    print(f"{split}: {n_traj} trajectories -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="/scratch/yl12783/datasets/meshgraphnets/deforming_plate")
    parser.add_argument("--out_root", default="/scratch/yl12783/datasets/meshgraphnets")
    parser.add_argument("--dataset_name", default="deforming_plate")
    parser.add_argument("--splits", nargs="+", default=["valid", "test", "train"])
    args = parser.parse_args()
    with open(os.path.join(args.in_dir, "meta.json")) as f:
        meta = json.load(f)
    for split in args.splits:
        convert_split(args.in_dir, args.out_root, args.dataset_name, split, meta)
