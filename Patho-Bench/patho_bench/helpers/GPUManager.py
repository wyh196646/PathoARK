import gc
import torch
import numpy as np
import time
import subprocess as sp

"""
GPUManager class for automatic load balancing of GPU memory.
"""

class GPUManager:

    @staticmethod
    def get_best_gpu(min_mb=1000, retry_interval=5, num_iters=100, verbose = False):
        '''
        Get the GPU with the most available memory, retrying if the memory is insufficient.

        Args:
            min_mb: int, minimum memory in MB required to use a GPU
            retry_interval: int, interval in minutes to wait before retrying
            num_iters: int, number of iterations to retry before giving up
            verbose: bool, flag for printing GPU memory information

        Returns:
            max_memory_gpu_index: int, index of GPU with the most available memory
        '''
        for _ in range(num_iters):
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

            if verbose:
                for i, mem in enumerate(memory_free_values):
                    print(f"GPU {i}: {mem / 1024:.2f} GB free")

            # Find the GPU with the most available memory
            max_memory_gpu_index = np.argmax(memory_free_values)
            max_memory_free = memory_free_values[max_memory_gpu_index]

            if max_memory_free >= min_mb:
                if verbose:
                    print(f"Using GPU {max_memory_gpu_index} with {max_memory_free / 1024:.2f} GB free")
                return max_memory_gpu_index
            
            print(f"No GPU with at least {min_mb} MB free, waiting {retry_interval} minutes before retrying...")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(retry_interval * 60)

        print(f"Could not find a GPU with at least {min_mb} MB free after {num_iters} iterations. Stopping.")
        exit()