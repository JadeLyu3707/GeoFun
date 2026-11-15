# @author: yl12783, Nov. 2026, New York
#
# Per-trajectory variation in deforming_plate: plate shape, obstacle size, obstacle motion, mesh size.
import numpy as np, h5py
B = "/scratch/yl12783/datasets/meshgraphnets/deforming_plate/data"
for split in ["train", "test"]:
    f = h5py.File(f"{B}/{split}/deforming_plate_{split}.h5", "r")
    R = []
    for k in f:
        g = f[k]; nt = g["node_type"][:]; c = g["cells"][:]; w0 = g["world_pos"][0]; w1 = g["world_pos"][-1]
        pl, ob = nt != 1, nt == 1
        pc = c[(nt[c] != 1).all(1)]; Q = w0[pc]
        v = np.abs(np.einsum("ij,ij->i", np.cross(Q[:, 1]-Q[:, 0], Q[:, 2]-Q[:, 0]), Q[:, 3]-Q[:, 0])).sum() / 6
        e = w0[pl].max(0) - w0[pl].min(0)
        oe = w0[ob].max(0) - w0[ob].min(0)
        disp = (w1[ob] - w0[ob]).mean(0); maxd = np.linalg.norm(g["world_pos"][:, ob].mean(1) - w0[ob].mean(0), axis=1).max()
        R.append([len(nt), e[0], e[1], e[2], v / np.prod(e), *oe, maxd, np.linalg.norm(disp), g["stress"][:].max()])
    R = np.array(R)
    names = ["n_nodes", "plate_x", "plate_y", "plate_thick", "plate_fill(vol/bbox)", "obst_dx", "obst_dy", "obst_dz",
             "obst_max_travel", "obst_end_offset", "max_stress"]
    print(f"\n== {split}: {len(R)} trajectories ==   (min / median / max)")
    for i, nm in enumerate(names):
        print(f"  {nm:22s} {R[:, i].min():10.4g} {np.median(R[:, i]):10.4g} {R[:, i].max():10.4g}")
    print(f"  plates with holes/notches (fill < 0.99): {(R[:, 4] < 0.99).sum()} / {len(R)}")
    print(f"  distinct plate thickness values: {np.unique(R[:, 3].round(4))[:12]}")
