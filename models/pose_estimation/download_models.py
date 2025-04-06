import os
import urllib.request
from pathlib import Path

def download_models():
    # Create models directory if it doesn't exist
    models_dir = Path("models/pose_estimation")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Model files to download
    model_files = {
        "pose_deploy_linevec.prototxt": "https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/5c5d96523ef917bd30301245fdc8343937cae48d/models/pose/mpi/pose_deploy_linevec.prototxtmodels/pose/mpi/pose_deploy_linevec.prototxt",
        "pose_iter_440000.caffemodel": "https://huggingface.co/camenduru/openpose/resolve/f5bb0c0a16060ac8b373472a5456c76bd68eb202/pose_iter_440000.caffemodel"
    }
    
    # Download each file
    for filename, url in model_files.items():
        file_path = models_dir / filename
        if not file_path.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
        else:
            print(f"{filename} already exists")

if __name__ == "__main__":
    download_models() 