# @author: yl12783, Nov. 2026, New York
#
# Adaptive (kNN density-weighted) Gaussian field: grid-search (d, k, c), choose tau per setting,
# compare against the fixed-width baseline, and plot the best setting.

import json
import logging
import os.path as osp

import hydra
import numpy as np
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data import PlateSnapshot
from models import AdaptiveGaussianPhi
from tuner import AdaptiveTuner, Evaler, VoxelGrid, build_ground_truth, format_adaptive_table
from utils import set_seed, set_out

logger = logging.getLogger("PHI")
logger.setLevel(level=logging.DEBUG)

KEYS = ("method", "d", "k", "c", "kind", "tau", "sigma_med", "loss", "extra", "missed", "iou",
        "out_mean", "out_med", "out_p95", "out_max", "field_in_median")


@hydra.main(version_base=None, config_path="configs", config_name="adaptive")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    out_folder = set_out(cfg)

    snapshot: PlateSnapshot = instantiate(cfg.data)
    print(snapshot.metadata)
    grid: VoxelGrid = instantiate(cfg.grid).build(snapshot.points)
    gt = build_ground_truth(grid, snapshot.tets, snapshot.is_obstacle_tet, snapshot.metadata.surface_area)
    logger.info(f"{grid}, voxelized volume {gt.volume:.4e} (true {snapshot.metadata.volume:.4e}), "
                f"surface area {gt.surface_area:.4f}")

    tuner: AdaptiveTuner = instantiate(cfg.tuner)
    rows, best = tuner.tune(snapshot.points, grid, gt)
    print(format_adaptive_table(rows))

    with open(osp.join(out_folder, "extended_config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    with open(osp.join(out_folder, "tune_results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(osp.join(out_folder, "best.yaml"), "w") as f:
        yaml.safe_dump({k: best.get(k) for k in KEYS}, f, sort_keys=False)
    logger.info(f"best (d={best['d']}, solid, min loss): k={best['k']}, c={best['c']}, tau={best['tau']:.4f}, "
                f"loss={best['loss']:.3f}")

    evaler: Evaler = instantiate(cfg.evaler, out_folder=out_folder)
    model = AdaptiveGaussianPhi(snapshot.points, k=best["k"], c=best["c"], d=best["d"], tau=best["tau"],
                                clamp=tuple(cfg.tuner.clamp))
    tag = f"d{best['d']}_k{best['k']}_c{best['c']}_tau{best['tau']:.4f}"
    evaler.EvalField(snapshot, grid, gt, model.field_on_grid(grid.axes), best["tau"],
                     f"adaptive d={best['d']}, k={best['k']}, c={best['c']}, tau={best['tau']:.4f}",
                     f"slices_{tag}.png", near_tol=float(np.median(model.sigma)))
    evaler.plot_point_stats(snapshot, model, f"points_{tag}.png")


if __name__ == "__main__":
    main()
