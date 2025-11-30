import torch
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset

def main():
    print("Loading Phikon Foundation Model...")
    # Load the Phikon Foundation Model
    # It extracts a 768-dimensional vector for every image
    try:
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        model = AutoModel.from_pretrained("owkin/phikon")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Move to GPU and Freeze
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    print("\nLoading PatchCamelyon dataset...")
    # Load from HuggingFace (no manual download needed)
    try:
        dataset = load_dataset("patch_camelyon")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Example: Get one image
    if 'train' in dataset:
        example_image = dataset['train'][0]['image']
        label = dataset['train'][0]['label'] # 1 = Metastasis, 0 = Normal
        
        print(f"\nExample loaded:")
        print(f"Image size: {example_image.size}")
        print(f"Label: {label} ({'Metastasis' if label == 1 else 'Normal'})")
    else:
        print("Train split not found in dataset.")

if __name__ == "__main__":
    main()

