import inspect
import json
import os
from patho_bench.config.JSONSaver import JSONsaver


class ConfigMixin:
    def get_config(self):
        """
        Retrieves configuration parameters for all arguments passed to the __init__ method.

        Returns:
            dict: A dictionary of init argument names and their configuration parameters.
        """
        # Get the signature of the __init__ method
        signature = inspect.signature(self.__init__)
        params = list(signature.parameters.keys())[1:] # Exclude 'self' from the parameters

        def recurse(value):
            if hasattr(value, 'get_config'):
                return value.get_config()
            elif isinstance(value, dict):
                return {k: recurse(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple, set)):
                return type(value)(recurse(v) for v in value)
            else:
                return value

        config = {}
        for param in params:
            value = getattr(self, param, None)
            config[param] = recurse(value)
        return config
    
    def save_config(self, saveto):
        '''
        Saves config parameters and hash to json file.

        Args:
            save_dir (str): Directory to save config to.
        '''
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        with open(saveto, 'w') as f:
            json.dump(self.get_config(), f, indent=4, cls=JSONsaver) # Use JSONsaver to serialize normally unserializable objects