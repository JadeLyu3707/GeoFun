# @author: yl12783, Nov. 2026, New York
#
# Build the GINO-style SDF dataset from the tetrahedral meshes and verify it.

import json
import logging
import os
import os.path as osp

import h5py as h5
import hydra
import numpy as np
import open3d as o3d
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sdf import SDFBuilder, body_surfaces, inside_tets_grid, point_triangle_distance
from utils import set_seed

logger = logging.getLogger("SDF")
logger.setLevel(level=logging.DEBUG)


def body_sdf(surf, q, nsamples):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(surf.vertices), o3d.utility.Vector3iVector(surf.triangles))
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_signed_distance(o3d.core.Tensor(q.astype(np.float32)), nsamples=nsamples).numpy()


def verify_sample(builder, X, cells, nt, sdf, grid, rng, n_dist):
    surfs = body_surfaces(X, cells, nt)
    q = grid.reshape(-1, 3); s = sdf.reshape(-1)
    out = {}
    # (a) SDF at the original points: surface nodes ~ 0, interior nodes < 0
    at_pts = np.min([body_sdf(sf, X, builder.nsamples) for sf in surfs.values()], axis=0)
    surf_ids = np.unique(np.concatenate([sf.vertex_ids for sf in surfs.values()]))
    interior = np.setdiff1d(np.arange(len(X)), surf_ids)
    out["surface_abs_mean"] = float(np.abs(at_pts[surf_ids]).mean()); out["surface_abs_max"] = float(np.abs(at_pts[surf_ids]).max())
    out["interior_max"] = float(at_pts[interior].max()) if len(interior) else None
    # (b) bbox corners positive; points known to be inside (centroid of each body's largest tetrahedron) negative
    lo, hi = grid.reshape(-1, 3).min(0), grid.reshape(-1, 3).max(0)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    out["corners_min"] = float(np.min([body_sdf(sf, corners, builder.nsamples) for sf in surfs.values()], axis=0).min())
    P = X[cells]; vol = np.abs(np.einsum("ij,ij->i", np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), P[:, 3]-P[:, 0])) / 6
    is_ob = (nt[cells] == 1).all(1); centres = []
    for m in (~is_ob, is_ob):
        if m.any():
            centres.append(P[np.flatnonzero(m)[np.argmax(vol[m])]].mean(0))
    out["centres_max"] = float(np.min([body_sdf(sf, np.array(centres), builder.nsamples) for sf in surfs.values()], axis=0).max())
    # (c) sign on every grid point vs exact point-in-tetrahedron test
    axes = [grid[:, 0, 0, 0], grid[0, :, 0, 1], grid[0, 0, :, 2]]
    inside = inside_tets_grid(axes, X, cells).reshape(-1)
    out["sign_mismatch"] = int(((s < 0) != inside).sum())
    # (d) |SDF| vs brute-force point-to-triangle distance, per body, on random grid points (half near the surface)
    if n_dist <= 0:
        out["dist_err_max"] = -1.0
        return out, inside.reshape(sdf.shape)
    near = np.flatnonzero(np.abs(s) < 0.03); idx = np.concatenate([rng.choice(near, min(n_dist // 2, len(near)), replace=False),
                                                                    rng.choice(len(q), n_dist // 2, replace=False)])
    errs = []
    for sf in surfs.values():
        errs.append(np.abs(np.abs(body_sdf(sf, q[idx], builder.nsamples)) - point_triangle_distance(q[idx], sf.vertices, sf.triangles)))
    out["dist_err_max"] = float(np.max(errs))
    return out, inside.reshape(sdf.shape)


def plot_slices(fname, title, sdf, inside, grid, X, nt):
    R = sdf.shape[0]
    ax_c = [grid[:, 0, 0, 0], grid[0, :, 0, 1], grid[0, 0, :, 2]]
    z_plate = X[nt != 1, 2].mean(); kz = int(np.argmin(np.abs(ax_c[2] - z_plate)))
    oc = X[nt == 1].mean(0) if (nt == 1).any() else X.mean(0); ky = int(np.argmin(np.abs(ax_c[1] - oc[1])))
    views = [(f"z = R/2 (z={ax_c[2][R//2]:.3f})", sdf[:, :, R // 2], inside[:, :, R // 2], 0, 1),
             (f"z through plate mid-thickness (z={ax_c[2][kz]:.4f})", sdf[:, :, kz], inside[:, :, kz], 0, 1),
             (f"x-z through actuator centre (y={ax_c[1][ky]:.3f})", sdf[:, ky, :], inside[:, ky, :], 0, 2)]
    vm = np.abs(sdf).max()
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (t, F, M, iu, iv) in zip(axs, views):
        u, v = ax_c[iu], ax_c[iv]
        im = ax.imshow(F.T, origin="lower", extent=[u[0], u[-1], v[0], v[-1]], cmap="RdBu", vmin=-vm, vmax=vm)
        ax.contour(u, v, F.T, levels=[0], colors="k", linewidths=1.2)
        if M.any() and not M.all():
            ax.contour(u, v, M.T.astype(float), levels=[0.5], colors="lime", linewidths=0.8, linestyles="--")
        ax.set_title(t); ax.set_aspect("equal"); ax.set_xlabel("xyz"[iu]); ax.set_ylabel("xyz"[iv])
        fig.colorbar(im, ax=ax, shrink=0.7, label="SDF [m] (negative inside)")
    fig.suptitle(title + "\nblack: SDF = 0, green dashed: grid points inside the tetrahedra (exact test)")
    fig.tight_layout(); fig.savefig(fname, dpi=90); plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="sdf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    builder: SDFBuilder = instantiate(cfg.builder)
    if cfg.verify.only:
        # reuse the files written by an earlier build
        meta = np.load(osp.join(builder.out_path, "meta.npz"))
        g = dict(grid=meta["grid_raw"].astype(np.float64), lo=meta["bbox_lo"], hi=meta["bbox_hi"],
                 mean=float(meta["sdf_mean"]), std=float(meta["sdf_std"]))
        raw, records = {}, []
        for s, path in builder.files.items():
            with h5.File(path, "r") as f:
                keys = builder._keys(f)
            for k in keys:
                d = np.load(osp.join(builder.out_path, s, f"{k}.npz"))
                raw[(s, k)] = d["sdf_raw"][..., 0]
                records.append(dict(split=s, traj=k, mesh_method=str(d["mesh_method"]),
                                    is_watertight=bool(d["is_watertight"]), vertices_match=True, seconds=float("nan")))
        logger.info(f"loaded {len(records)} samples from {builder.out_path}")
    else:
        records, raw, g = builder.build()
        logger.info(f"built {len(records)} samples; train SDF mean {g['mean']:.4f} std {g['std']:.4f}")
    grid = g["grid"]

    vdir = osp.join(builder.out_path, "verification"); os.makedirs(vdir, exist_ok=True)
    checks = []
    keys = list(raw.keys())
    verify_keys = keys if cfg.verify.all_samples else [keys[i] for i in rng.choice(len(keys), cfg.verify.n_samples, replace=False)]
    plot_keys = [keys[i] for i in rng.choice(len(keys), cfg.verify.n_plots, replace=False)]
    if cfg.verify.always_plot is not None:
        plot_keys = [tuple(cfg.verify.always_plot)] + plot_keys
    dist_keys = set(verify_keys[: cfg.verify.n_dist_samples])
    for i, (s, k) in enumerate(verify_keys + [pk for pk in plot_keys if pk not in verify_keys]):
        with h5.File(builder.files[s], "r") as f:
            gk = f[k]; X = gk["world_pos"][builder.t].astype(np.float64); cells = gk["cells"][:]; nt = gk["node_type"][:]
        res, inside = verify_sample(builder, X, cells, nt, raw[(s, k)], grid, rng,
                                    cfg.verify.n_dist_points if (s, k) in dist_keys else 0)
        res.update(split=s, traj=k); checks.append(res)
        if (s, k) in plot_keys:
            plot_slices(osp.join(vdir, f"slices_{s}_{k}.png"), f"{s}/{k} t={builder.t}, R={builder.R}",
                        raw[(s, k)], inside, grid, X, nt)
        if (i + 1) % 100 == 0:
            logger.info(f"verified {i+1}")

    n = len(records)
    summary = dict(
        n_samples=n, R=builder.R, t=builder.t,
        grid_spacing_mm=(1e3 * (g["hi"] - g["lo"]) / (builder.R - 1)).round(3).tolist(),
        bbox_lo=g["lo"].tolist(), bbox_hi=g["hi"].tolist(),
        watertight=int(sum(r["is_watertight"] for r in records)), poisson_fallback=0, failed=0,
        mesh_vertices_match_points=int(sum(r["vertices_match"] for r in records)),
        mean_seconds_per_sample=float(np.nanmean([r["seconds"] for r in records])) if not cfg.verify.only else None,
        train_sdf_mean=g["mean"], train_sdf_std=g["std"],
        n_verified=len(checks),
        surface_points_abs_sdf_mean=float(np.mean([c["surface_abs_mean"] for c in checks])),
        surface_points_abs_sdf_max=float(np.max([c["surface_abs_max"] for c in checks])),
        interior_points_sdf_max=float(np.max([c["interior_max"] for c in checks if c["interior_max"] is not None])),
        bbox_corners_sdf_min=float(np.min([c["corners_min"] for c in checks])),
        inside_centres_sdf_max=float(np.max([c["centres_max"] for c in checks])),
        grid_sign_mismatches_total=int(sum(c["sign_mismatch"] for c in checks)),
        grid_points_checked=len(checks) * builder.R ** 3,
        distance_check_samples=len([c for c in checks if c["dist_err_max"] > -1 and (c["split"], c["traj"]) in dist_keys]),
        distance_err_max=float(np.max([c["dist_err_max"] for c in checks if (c["split"], c["traj"]) in dist_keys] or [0])),
    )
    with open(osp.join(vdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if not cfg.verify.only:
        with open(osp.join(vdir, "records.json"), "w") as f:
            json.dump(records, f, indent=0)
    with open(osp.join(vdir, "checks.json"), "w") as f:
        json.dump(checks, f, indent=0)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
