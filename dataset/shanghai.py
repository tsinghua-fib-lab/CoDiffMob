import pickle
import sys
from typing import Callable, Dict, Optional

import numpy as np
import pyproj
import torch
from torch.utils.data import Dataset

TIME_INTERVAL = 1800
LENGTH = 86400
GRID_SIZE = 1000
INDEX2XY = pickle.load(
    open(
        "/path/to/idx2xy.pkl",
        "rb",
    )
)
INDEX2XY = np.array([INDEX2XY[idx] for idx in range(len(INDEX2XY))])
MINX, MINY = INDEX2XY.min(axis=0)
MAXX, MAXY = INDEX2XY.max(axis=0)
GRID_NUM_X = int((MAXX - MINX) / GRID_SIZE) + 1
GRID_NUM_Y = int((MAXY - MINY) / GRID_SIZE) + 1
LONLAT_MEAN = np.array([121.4554, 31.1791])
LONLAT_STD = np.array([0.1527, 0.1323])


def xy2grid(xy: np.ndarray) -> np.ndarray:
    x_grid = ((xy[:, 0] - MINX) / GRID_SIZE).astype(int)
    x_grid = np.clip(x_grid, 0, GRID_NUM_X - 1)
    y_grid = ((MAXY - xy[:, 1]) / GRID_SIZE).astype(int)
    y_grid = np.clip(y_grid, 0, GRID_NUM_Y - 1)
    return np.stack([y_grid, x_grid], axis=1)


def norm_xy(xy: np.ndarray) -> np.ndarray:
    x = (xy[:, 0] - MINX) / (MAXX - MINX)
    y = (xy[:, 1] - MINY) / (MAXY - MINY)
    return np.stack([x, y], axis=1)


class ShanghaiDataset(Dataset):
    def __init__(
        self,
        root: str,
        target: str = "emb",
        length: Optional[int] = None,
    ):
        super().__init__()
        self.data = pickle.load(open(root, "rb"))
        self.target = target
        self.emb = None
        if self.target == "emb":
            raise NotImplementedError
        self.length = length
        self.proj = pyproj.Proj(
            "+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )

    def __len__(self) -> int:
        if self.length is not None:
            assert self.length <= len(self.data)
            return self.length
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        target = self.preprocess(self.data[index])
        t = torch.arange(0, LENGTH, TIME_INTERVAL).view(-1).long()
        loc = self.data[index]["loc_id"]
        xy = self.data[index]["xy"]
        return {
            "x": target,
            "x_t": t,
            "x_loc": torch.tensor(loc, dtype=torch.long),
            "x_pos": torch.tensor(xy, dtype=torch.float32),
        }

    def preprocess(self, data: Dict) -> torch.Tensor:
        if self.target == "lonlat":
            xy = data["xy"]
            lon, lat = self.proj(xy[:, 0], xy[:, 1], inverse=True)
            lonlat = np.stack([lon, lat], axis=1)
            lonlat = (lonlat - LONLAT_MEAN) / LONLAT_STD
            return torch.tensor(lonlat, dtype=torch.float32)
        elif self.target == "emb":
            index = data["loc_id"]
            return self.emb[index].clone().detach()
        elif self.target == "loc":
            return torch.tensor(data["loc_id"], dtype=torch.long)
        elif self.target == "xy":
            xy = data["xy"]
            return torch.tensor(norm_xy(xy), dtype=torch.float32)
        elif self.target == "grid":
            xy = data["xy"]
            return torch.tensor(xy2grid(xy), dtype=torch.long)
        else:
            raise ValueError(f"Unknown target: {self.target}")
