from pathlib import Path
import os
import sys

# set env variable to use the correct GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# get current file path
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = Path(current_file_path).parent

# add source folder to path
sys.path.insert(0, str(project_root_path / "denoising-diffusion-pytorch"))

if __name__ == "__main__":
    print(sys.path)