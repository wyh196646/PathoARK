from patho_bench.datasets.BaseDataset import BaseDataset

"""
CombinedDataset() is a simple dataset wrapper class that combines multiple child datasets into a single dataset.
"""


class CombinedDataset(BaseDataset):
    def __init__(self, child_datasets):
        '''
        Args:
            child_datasets (dict[Dataset]): Dict of child datasets
        '''
        super().__init__(list(child_datasets.values())[0].split) # Use the split of the first child dataset

        # Assert all child datasets have the same IDs
        for dataset_name, dataset in child_datasets.items():
            assert dataset.ids == self.ids, f'All child datasets must have the same IDs. Dataset {dataset_name} has different IDs.'
        self.child_datasets = child_datasets

    def __getitem__(self, idx):
        '''
        Args:
            idx (int or str): Index of sample to return or sample ID
        '''
        batch = {'ids': self.ids[idx] if isinstance(idx, int) else idx}
        for dataset_name, dataset in self.child_datasets.items():
            batch[dataset_name] = dataset[idx]
            assert dataset[idx]['id'] == self.ids[idx], f'Sample {idx} in dataset {dataset_name} has different ID than expected: {dataset[idx]["id"]} != {self.ids[idx]}'

        return batch