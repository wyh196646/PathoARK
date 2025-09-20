import os
import torch
import h5py
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class SlidePretrainDataset(Dataset):
    """
    Dataset for supervised pretraining on slide-level embeddings.
    Returns two augmented views (by shuffling/subsampling tiles) and the label vector.

    Arguments:
    ----------
    data_df: pd.DataFrame with columns including slide_key, split_key, label(s)
    root_path: folder containing .h5 or .pt per slide
    splits: list of IDs selected for this subset
    task_config: dict with 'setting' and 'label_dict' or multi-label columns
    view_max_tiles: cap tiles per view; if None, use task_config.max_tiles
    shuffle_tiles: shuffle tiles order to create different views
    subsample_ratio: float in (0,1], randomly sample that proportion of tiles per view
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        root_path: str,
        splits: list,
        task_config: dict,
        slide_key: str = "slide_id",
        split_key: str = "pat_id",
        view_max_tiles: int | None = None,
        shuffle_tiles: bool = True,
        subsample_ratio: float = 1.0,
    ):
        super().__init__()
        self.root_path = root_path
        self.slide_key = slide_key
        self.split_key = split_key
        self.task_cfg = task_config
        self.shuffle_tiles = shuffle_tiles
        self.subsample_ratio = subsample_ratio
        self.max_tiles = view_max_tiles or task_config.get("max_tiles", 1000)

        valid = self._get_valid_slides(root_path, data_df[slide_key].values)
        data_df = data_df[data_df[slide_key].isin(valid)]
        self._setup_data(data_df, splits, task_config.get("setting", "multi_class"))

    def _get_valid_slides(self, root_path: str, slides: list) -> list:
        valids = []
        for s in slides:
            fname = s.replace(".svs", "") + (".pt" if "pt_files" in root_path.split("/")[-1] else ".h5")
            if os.path.exists(os.path.join(root_path, fname)):
                valids.append(s)
        return valids

    def _setup_data(self, df: pd.DataFrame, splits: list, task: str):
        if task in ["multi_class", "binary"]:
            label_dict = self.task_cfg.get("label_dict", {})
            assert label_dict, "No label_dict in task_config"
            assert "label" in df.columns, "label column missing"
            df = df[df[self.split_key].isin(splits)].copy()
            df["label"] = df["label"].map(label_dict)
            self.n_classes = len(label_dict)
            self.images = df[self.slide_key].tolist()
            self.labels = df[["label"]].to_numpy().astype(int)
        elif task == "multi_label":
            label_dict = self.task_cfg.get("label_dict", {})
            assert label_dict, "No label_dict in task_config"
            keys = sorted(label_dict.keys(), key=lambda x: label_dict[x])
            df = df[df[self.split_key].isin(splits)].copy()
            self.n_classes = len(label_dict)
            self.images = df[self.slide_key].tolist()
            self.labels = df[keys].to_numpy().astype(int)
        else:
            raise ValueError(f"Invalid task: {task}")

    def __len__(self):
        return len(self.images)

    def _read_assets(self, path: str):
        if path.endswith(".pt"):
            images = torch.load(path)
            coords = torch.zeros(images.size(0), 2)
            return images, coords
        assets = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                assets[k] = f[k][:]
        images = torch.from_numpy(assets["features"])
        coords = torch.from_numpy(assets["coords"]) if "coords" in assets else torch.zeros(images.size(0), 2)
        return images, coords

    def _augment(self, images: torch.Tensor, coords: torch.Tensor):
        n = images.size(0)
        idx = torch.arange(n)
        if self.shuffle_tiles:
            idx = torch.randperm(n)
        if self.subsample_ratio < 1.0:
            k = max(1, int(n * self.subsample_ratio))
            idx = idx[:k]
        images = images[idx]
        coords = coords[idx]
        if images.size(0) > self.max_tiles:
            images = images[: self.max_tiles]
            coords = coords[: self.max_tiles]
        return images, coords

    def __getitem__(self, idx):
        sid = self.images[idx]
        path = os.path.join(
            self.root_path,
            sid.replace(".svs", "") + (".pt" if "pt_files" in self.root_path.split("/")[-1] else ".h5"),
        )
        images, coords = self._read_assets(path)
        v1_imgs, v1_coords = self._augment(images, coords)
        v2_imgs, v2_coords = self._augment(images, coords)
        label = torch.from_numpy(self.labels[idx])
        return {
            "imgs1": v1_imgs,
            "coords1": v1_coords,
            "imgs2": v2_imgs,
            "coords2": v2_coords,
            "labels": label,
            "slide_id": sid,
        }
