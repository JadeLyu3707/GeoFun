# @author: yl12783, Nov. 2026, New York
#
# Evaluate a fixed (sigma, tau). If model.tau is null, (sigma, tau) are read from
# experiments/<experiment>/best.yaml written by tune.py.

import logging
import os.path as osp

import hydra
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import PlateSnapshot
from tuner import Evaler, VoxelGrid, build_ground_truth
from utils import set_seed, set_out

logger = logging.getLogger("PHI")
logger.setLevel(level=logging.DEBUG)


@hydra.main(version_base=None, config_path="configs", config_name="phi")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    out_folder = set_out(cfg)

    sigma, tau = cfg.model.sigma, cfg.model.tau
    if tau is None:
        best_file = osp.join(out_folder, "best.yaml")
        assert osp.isfile(best_file), f"model.tau is null and no {best_file}; run tune.py or set model.tau"
        best = yaml.safe_load(open(best_file))
        sigma, tau = best["sigma"], best["tau"]
    logger.info(f"Evaluate sigma={sigma}, tau={tau}")

    snapshot: PlateSnapshot = instantiate(cfg.data)
    grid: VoxelGrid = instantiate(cfg.grid).build(snapshot.points)
    gt = build_ground_truth(grid, snapshot.tets, snapshot.is_obstacle_tet,
                            snapshot.metadata.surface_area)

    evaler: Evaler = instantiate(cfg.evaler, out_folder=out_folder)
    m = evaler.Eval(snapshot, grid, gt, sigma, tau)
    print(yaml.safe_dump({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}, sort_keys=False))
    with open(osp.join(out_folder, f"eval_sigma{sigma}_tau{tau:.5f}.yaml"), "w") as f:
        yaml.safe_dump(m, f, sort_keys=False)


if __name__ == "__main__":
    main()
