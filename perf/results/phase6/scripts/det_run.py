import os
import runpy
import sys

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.path.insert(0, os.getcwd())
sys.argv = ["infer.py"] + sys.argv[1:]
runpy.run_path("infer.py", run_name="__main__")
