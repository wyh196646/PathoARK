import os
import pandas as pd

from ark_params import get_ark_pretrain_params
from ark_pretrain import ark_pretrain_run
from datasets.pretrain_dataset import SlidePretrainDataset


def build_dataset_from_cfg(d, split: str):
    # d has keys: csv, root_path, split_dir, task_cfg_path, slide_key, split_key, max_tiles, shuffle_tiles, subsample_ratio, num_classes
    csv_path = d["csv"] if isinstance(d["csv"], str) else d["csv"][split]
    df = pd.read_csv(csv_path)
    with open(d["task_cfg_path"], 'r') as f:
        import yaml
        task_cfg = yaml.safe_load(f)
    # choose splits list from precomputed CSVs under split_dir
    split_csv = os.path.join(d["split_dir"], f"{split}.csv")
    if os.path.exists(split_csv):
        split_ids = pd.read_csv(split_csv)[d.get("split_key", "pat_id")].tolist()
    else:
        # if not provided, take unique ids present (not ideal but works)
        split_ids = df[d.get("split_key", "pat_id")].drop_duplicates().tolist()

    ds = SlidePretrainDataset(
        data_df=df,
        root_path=d["root_path"],
        splits=split_ids,
        task_config=task_cfg,
        slide_key=d.get("slide_key", "slide_id"),
        split_key=d.get("split_key", "pat_id"),
        view_max_tiles=d.get("max_tiles", task_cfg.get("max_tiles", 1000)),
        shuffle_tiles=d.get("shuffle_tiles", True),
        subsample_ratio=d.get("subsample_ratio", 1.0),
    )
    # choose criterion based on task setting
    import torch as _torch
    setting = task_cfg.get('setting', 'multi_class')
    if setting == 'multi_label':
        crit = _torch.nn.BCEWithLogitsLoss()
    else:
        crit = _torch.nn.CrossEntropyLoss()
    return ds, crit


if __name__ == '__main__':
    args = get_ark_pretrain_params()
    ckpt = ark_pretrain_run(args, build_dataset_fn=build_dataset_from_cfg, datasets_yaml=args.datasets_yaml)
    print(f"Pretraining finished. Best checkpoint: {ckpt}")
