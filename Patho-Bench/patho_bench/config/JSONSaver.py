import json
import torch
import numpy as np

class JSONsaver(json.JSONEncoder):
    '''
    Custom encoder for JSON to handle normally unserializable stuff
    
    Used in:
        ConfigMixin.save_config(): Save config parameters to json file.
    '''
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, range):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return str(obj)
        elif isinstance(obj, torch.dtype):
            return str(obj)
        elif callable(obj):
            if hasattr(obj, '__name__'):
                if obj.__name__ == '<lambda>':
                    return f'CALLABLE.{id(obj)}' # Unique identifier for lambda functions
                else:   
                    return f'CALLABLE.{obj.__name__}'
            else:
                return f'CALLABLE.{str(obj)}'
        else:
            return super().default(obj)