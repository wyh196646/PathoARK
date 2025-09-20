import os
import argparse
import time
import sys; sys.path.append('..')
from patho_bench.Runner import Runner

"""
Continuously prints the progress of an experiment until all tmux panes are done.
"""

# Argparse a single argument: experiment ID
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', type=str, help='Directory containing the experiment to monitor.')
parser.add_argument('--splits_root', type=str, default='../artifacts/splits', help='Root directory for the splits')
args = parser.parse_args()

if __name__ == '__main__':
    Runner.monitor_progress(args.experiment_dir)
    import time; time.sleep(2)
    Runner.collect_results(args.experiment_dir, args.splits_root)