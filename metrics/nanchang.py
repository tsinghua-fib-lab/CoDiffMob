import sys

sys.path.append("/data1/zhangyuheng/urban-mobility")

import pickle
from typing import List, Tuple

import numpy as np
import pyproj
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon

INDEX2XY = pickle.load(
    open(
        "/path/to/idx2xy.pkl",
        "rb",
    )
)
INDEX2XY = np.array([INDEX2XY[idx] for idx in range(len(INDEX2XY))])
MINX, MINY = INDEX2XY.min(axis=0)
MAXX, MAXY = INDEX2XY.max(axis=0)
INDEX2XY_NORM = np.stack(
    [
        (INDEX2XY[:, 0] - MINX) / (MAXX - MINX),
        (INDEX2XY[:, 1] - MINY) / (MAXY - MINY),
    ],
    axis=1,
)
proj = pyproj.Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lon, lat = proj(INDEX2XY[:, 0], INDEX2XY[:, 1], inverse=True)
INDEX2LONLAT = np.stack([lon, lat], axis=1)
LONLAT_MEAN = np.array([115.89902396, 28.69024124])
LONLAT_STD = np.array([0.84300954, 0.77130944])

GRID_SIZE = 1000
GRID_NUM_X = int((MAXX - MINX) / GRID_SIZE) + 1
GRID_NUM_Y = int((MAXY - MINY) / GRID_SIZE) + 1
INDEX2GRID = ((MAXY - INDEX2XY[:, 1]) / GRID_SIZE).astype(int) * GRID_NUM_X + (
    (INDEX2XY[:, 0] - MINX) / GRID_SIZE
).astype(int)


def xy2grid(xy: np.ndarray) -> np.ndarray:
    x_grid = ((xy[:, 0] - MINX) / GRID_SIZE).astype(int)
    y_grid = ((MAXY - xy[:, 1]) / GRID_SIZE).astype(int)
    return np.stack([y_grid, x_grid], axis=1)


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * jensenshannon(p, m) + 0.5 * jensenshannon(q, m)


def ks_test(real: List[float], gen: List[float]) -> float:
    res = stats.ks_2samp(real, gen)
    return res.statistic


def merge_same_elements(lst):
    merged = []
    for i in range(len(lst)):
        if i == 0 or lst[i] != lst[i - 1]:
            merged.append(lst[i])
    return merged


def travel_distance(idx: torch.Tensor, index2xy: np.ndarray = INDEX2XY) -> List[float]:
    distances = []
    for i in range(idx.shape[0]):
        idx_i = idx[i]
        merged = INDEX2XY[merge_same_elements(idx_i)]
        distances.extend(np.linalg.norm(merged[1:] - merged[:-1], axis=1).tolist())
    return distances


def gyration_radius(idx: torch.Tensor, index2xy: np.ndarray = INDEX2XY) -> List[float]:
    radiuses = []
    for i in range(idx.shape[0]):
        xy_i = index2xy[idx[i]]
        center = np.mean(xy_i, axis=0)
        n = xy_i.shape[0]
        radiuses.append(np.sqrt(np.sum(np.linalg.norm(xy_i - center, axis=1) ** 2) / n))
    return radiuses


def duration(idx: torch.Tensor, index2xy: np.ndarray = INDEX2XY) -> List[int]:
    idx = torch.from_numpy(INDEX2GRID[idx])
    durations = []
    for i in range(idx.shape[0]):
        idx_i = idx[i]
        diff = idx_i[1:] - idx_i[:-1]
        where = torch.where(diff != 0)[0]
        if len(where) == 0:
            durations.append(len(idx_i))
            continue
        durations.append(where[0].item())
        durations.extend((where[1:] - where[:-1]).tolist())
    return durations


def daily_loc(idx: torch.Tensor, index2xy: np.ndarray = INDEX2XY) -> List[int]:
    idx = torch.from_numpy(INDEX2GRID[idx])
    daily_locs = []
    for i in range(idx.shape[0]):
        idx_i = idx[i]
        daily_locs.append(len(torch.unique(idx_i)))
    return daily_locs


def complete_transition_matrix(idx: torch.Tensor) -> torch.Tensor:
    idx = INDEX2GRID[idx]
    matrix = torch.zeros(INDEX2GRID.max() + 1, INDEX2GRID.max() + 1)
    # matrix = torch.zeros(len(INDEX2LONLAT), len(INDEX2LONLAT))
    interval = (2 * 60) // 30
    for i in range(idx.shape[0]):
        idx_i = idx[i]
        idx_i = np.concatenate([idx_i[::interval], [idx_i[-1]]], axis=0)
        for j in range(idx_i.shape[0] - 1):
            if idx_i[j] == idx_i[j + 1]:
                continue
            matrix[idx_i[j], idx_i[j + 1]] += 1
    return matrix
