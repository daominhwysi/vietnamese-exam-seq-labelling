import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add local path
sys.path.append(str(Path(__file__).parent.parent))


def upload_dataset(token=None, repo_id=None):
    # Load environment variables from .env file
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

    if not token:
        token = os.getenv("HF_TOKEN")
    if not token:
        print(
            "Error: HF_TOKEN not found in environment. Please make sure it is defined in your '.env' file."
        )
        sys.exit(1)

    if not repo_id:
        repo_id = "daominhwysi/synthetic-seq-labelling-vi-exam-v2"

    # Try importing huggingface_hub
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print(
            "Error: 'huggingface_hub' is not installed. Installing it inside the active Pixi environment..."
        )
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Ensuring repository '{repo_id}' exists on Hugging Face Hub...")
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
        print("Repository is ready.")
    except Exception as e:
        print(f"Notice during repository creation: {e}")

    print(f"Uploading files from 'output/dataset' to dataset repo '{repo_id}'...")
    try:
        api.upload_folder(
            folder_path="output/dataset",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload synthetic sequence labelling dataset for Vietnamese exams",
        )
        print(
            f"\nSuccess! Dataset uploaded to: https://huggingface.co/datasets/{repo_id}"
        )
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    upload_dataset()
