from patho_bench.datasets.BaseDataset import BaseDataset
import torch

"""
LabelDataset loads one or more labels per sample.
"""

class LabelDataset(BaseDataset):
    def __init__(self, split, task_names, dtype, extra_attrs = None):
        super().__init__(split)
        self.task_names = task_names
        self.extra_attrs = extra_attrs
        self.dtype = dtype
        
    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        if self.dtype == 'float':
            dtype = torch.float32
            labels = {task: torch.tensor(float(self.data[sample_id]['labels'][task]), dtype=dtype) for task in self.task_names}
        elif self.dtype == 'int':
            dtype = torch.int64
            labels = {task: torch.tensor(int(self.data[sample_id]['labels'][task]), dtype=dtype) for task in self.task_names}
        elif self.dtype == 'str':
            dtype = torch.int64
            raise NotImplementedError('Label dtype "str" not yet implemented')
        else:
            raise ValueError(f'Label dtype {self.dtype} not recognized, must be "float", "int", or "str"')
        
        if self.extra_attrs:
            labels['extra_attrs'] = {}
            for attr in self.extra_attrs:
                labels['extra_attrs'][attr] = torch.tensor(self.data[sample_id][attr][0]) # HACK: Current implementation of Split keeps extra_attrs for each slide in a patient with multiple slides. We assume that every slide has the same extra_attrs, so here we just take the first slide's extra_attrs.

        labels['id'] = sample_id
        return labels
        
        
