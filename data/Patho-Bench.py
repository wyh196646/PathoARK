import os
import datasets
from datasets import Features, Value
from huggingface_hub import snapshot_download
import glob
import yaml


class PathoBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        
        # Extract task_in_dataset and dataset_to_download from kwargs
        self.task_in_dataset = kwargs.pop("task_in_dataset", None)
        self.dataset_to_download = kwargs.pop("dataset_to_download", None)
        self.force_download = kwargs.pop("force_download", True)
        
        # Set default values for task_in_dataset and dataset_to_download
        if self.dataset_to_download is None and self.task_in_dataset is None:
            # If neither are provided, default both to '*'
            self.dataset_to_download = '*'
            self.task_in_dataset = '*'
        elif self.dataset_to_download is None and self.task_in_dataset is not None:
            # If task_in_dataset is provided but dataset_to_download is not, raise an error
            raise AssertionError("Dataset needs to be defined for the task_in_dataset provided.")
        elif self.dataset_to_download is not None and self.task_in_dataset is None:
            # If dataset_to_download is provided but task_in_dataset is not, default task_in_dataset to '*'
            self.task_in_dataset = '*'
            
        super().__init__(**kwargs)


class PathoBenchDataset(datasets.GeneratorBasedBuilder):
    """
    Downloads only the .tsv and .yaml files needed to construct the dataset.
    Excludes .png images so they don't break the builder.
    """
    BUILDER_CONFIGS = [
        PathoBenchConfig(name="custom_config", version="1.0.0", description="PathoBench config")
    ]
    BUILDER_CONFIG_CLASS = PathoBenchConfig

    def _info(self):
        return datasets.DatasetInfo(
            description="PathoBench: collection of canonical computational pathology tasks",
            homepage="https://github.com/mahmoodlab/patho-bench",
            license="CC BY-NC-SA 4.0 Deed",
            features=Features({
                'path': Value('string')
            })
        )

    def _split_generators(self, dl_manager):
        repo_id = "MahmoodLab/patho-bench"
        dataset_to_download = self.config.dataset_to_download
        local_dir = self._cache_dir_root
        force_download = self.config.force_download
        task_in_dataset = self.config.task_in_dataset

        # Ensure the base local directory exists
        os.makedirs(local_dir, exist_ok=True)

        # 1) Download the top-level available_splits.yaml
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["available_splits.yaml"],  # only this file
            repo_type="dataset",
            local_dir=local_dir,
            force_download=force_download,
        )

        # Read available splits
        with open(os.path.join(local_dir, "available_splits.yaml"), 'r') as file:
            available_splits = yaml.safe_load(file)
        
        # Basic validation
        if dataset_to_download != "*":
            assert dataset_to_download in available_splits, (
                f"{dataset_to_download} was not found. "
                f"Available splits: {list(available_splits.keys())}"
            )
            if task_in_dataset != "*":
                assert task_in_dataset in available_splits[dataset_to_download], (
                    f"{task_in_dataset} was not found in {dataset_to_download}. "
                    f"Available tasks: {available_splits[dataset_to_download]}"
                )

        # 2) Decide what to allow based on dataset/task
        #
        # We only want .tsv and the relevant .yaml files (like about.yaml, config.yaml).
        # That way, we skip .png images which can cause issues or be large in LFS.
        if dataset_to_download == "*":
            # Download every dataset subfolder's .tsv and about.yaml/config.yaml
            allow_patterns = [
                "**/*.tsv",             # All tsv splits
                "**/about.yaml",        # The about files
                "**/config.yaml",       # The config files
                "available_splits.yaml" # Already downloaded, but no harm
            ]
        else:
            if task_in_dataset == "*":
                allow_patterns = [
                    f"{dataset_to_download}/**/*.tsv",
                    f"{dataset_to_download}/**/about.yaml",
                    f"{dataset_to_download}/**/config.yaml",
                    "available_splits.yaml"
                ]
            else:
                allow_patterns = [
                    f"{dataset_to_download}/{task_in_dataset}/*.tsv",
                    f"{dataset_to_download}/{task_in_dataset}/config.yaml",
                    f"{dataset_to_download}/about.yaml",
                    "available_splits.yaml"
                ]

        # 3) Download the requested patterns
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            repo_type="dataset",
            local_dir=local_dir,
            force_download=force_download,
        )
        
        # 4) Locate all .tsv files to pass to _generate_examples
        search_pattern = os.path.join(local_dir, '**', '*.tsv')
        all_tsv_splits = glob.glob(search_pattern, recursive=True)
        
        return [
            datasets.SplitGenerator(
                name="full",
                gen_kwargs={"filepath": all_tsv_splits},
            )
        ]

    def _generate_examples(self, filepath):
        idx = 0
        for file in filepath:
            yield idx, {
                'path': file
            }
            idx += 1