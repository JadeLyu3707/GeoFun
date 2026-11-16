# @author: yl12783, Nov. 2026, New York
#
# Grid-search sigma (tau in closed form per sigma), report the error table, and plot the best setting.

import json
import logging
import os.path as osp

import hydra
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data import PlateSnapshot
from tuner import Tuner, Evaler, VoxelGrid, build_ground_truth, format_table
from utils import set_seed, set_out

logger = logging.getLogger("PHI")
logger.setLevel(level=logging.DEBUG)


@hydra.main(version_base=None, config_path="configs", config_name="phi")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    out_folder = set_out(cfg)

    logger.info(f"Instantiate data {cfg.data._target_}")
    snapshot: PlateSnapshot = instantiate(cfg.data)
    print(snapshot.metadata)

    logger.info(f"Instantiate grid {cfg.grid._target_}")
    grid: VoxelGrid = instantiate(cfg.grid).build(snapshot.points)
    gt = build_ground_truth(grid, snapshot.tets, snapshot.is_obstacle_tet,
                            snapshot.metadata.surface_area)
    logger.info(f"{grid}, voxelized volume {gt.volume:.4e} (true {snapshot.metadata.volume:.4e})")

    logger.info(f"Instantiate tuner {cfg.tuner._target_}")
    tuner: Tuner = instantiate(cfg.tuner)
    rows, best = tuner.tune(snapshot.points, grid, gt)
    print(format_table(rows))

    with open(osp.join(out_folder, "extended_config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    with open(osp.join(out_folder, "tune_results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(osp.join(out_folder, "best.yaml"), "w") as f:
        yaml.safe_dump({k: best[k] for k in ("sigma", "tau", "loss", "extra", "missed", "iou",
                                             "out_med", "out_p95", "out_max")}, f, sort_keys=False)
    logger.info(f"best (solid, min loss): sigma={best['sigma']}, tau={best['tau']:.5f}, loss={best['loss']:.3f}")

    evaler: Evaler = instantiate(cfg.evaler, out_folder=out_folder)
    evaler.Eval(snapshot, grid, gt, best["sigma"], best["tau"])


if __name__ == "__main__":
    main()
