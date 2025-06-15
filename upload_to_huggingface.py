import argparse
import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

def upload_to_huggingface(json_file: str, repo_name: str, hf_token: str = None, private: bool = False):
    """
    Upload a JSON file to a new Hugging Face dataset repository.

    Args:
        json_file (str): Path to the JSON file containing the dataset.
        repo_name (str): Name of the Hugging Face repository to create (e.g., 'your-username/your-dataset-name').
        hf_token (str, optional): Hugging Face API token. If not provided, it will try to use a saved token.
        private (bool, optional): Whether to make the dataset private. Defaults to False.
    """
    # Authenticate with Hugging Face
    if hf_token:
        print("Using provided Hugging Face token.")
        HfFolder.save_token(hf_token)
    elif not os.getenv('HUGGING_FACE_HUB_TOKEN'):
        print("Hugging Face token not found. Please provide it via the --hf_token argument or set the HUGGING_FACE_HUB_TOKEN environment variable.")
        return

    # Load the dataset from the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to Hugging Face Dataset object
    hf_dataset = Dataset.from_list(data)

    # Create a DatasetDict (optional but good practice)
    dataset_dict = DatasetDict({
        'train': hf_dataset
    })

    print(f"Loaded {len(hf_dataset)} examples.")

    # Push to Hugging Face Hub
    try:
        print(f"Uploading dataset to Hugging Face Hub: {repo_name}")
        dataset_dict.push_to_hub(repo_name, private=private)
        print("\nUpload successful!")
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure that the repository name is in the format 'username/dataset-name' and that your token has 'write' permissions.")

if __name__ == "__main__":

    upload_to_huggingface("_WeLoveYou__conversation_training_data.json", "ThomasTheMaker/WeLoveYou")
