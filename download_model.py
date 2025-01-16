from huggingface_hub import hf_hub_download
import os
from pathlib import Path
from tqdm import tqdm

def download_model(
    repo_id: str = "TheBloke/Llama-2-13B-chat-GGUF" , # Updated repo
    filename: str = "llama-2-13b-chat.Q4_K_M.gguf",   # Updated filename
    local_dir: str = "models"
):
    """
    Download a model from Hugging Face Hub with progress bar
    """
    os.makedirs(local_dir, exist_ok=True)
    
    
    print(f"Downloading {filename} from {repo_id}...")
    print("This might take a while depending on your internet speed...")
    
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"\nModel downloaded successfully to: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Using a smaller, verified model
        model_path = download_model()
    except Exception as e:
        print("\nIf you continue to have issues, try these steps:")
        