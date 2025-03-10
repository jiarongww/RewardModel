from cProfile import label
import os
import json
import numpy as np
from fairscale.nn.model_parallel.random import checkpoint
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import requests
from clint.textui import progress
import huggingface_hub
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers.models.align.convert_align_tf_to_hf import preprocess
from torchvision.transforms.functional import to_pil_image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer, tokenize
from hpsv2.src.training.train import calc_ImageReward, inversion_score
from hpsv2.src.training.data import ImageRewardDataset, collate_rank, RankingDataset
from hpsv2.utils import root_path, hps_version_map
from ImageReward.ImageReward import ImageReward, BLIPScore, CLIPScore, AestheticScore
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image
import torch
import torch.nn.functional as F

model_dict = {}
model_name = "ViT-H-14"
precision = 'amp'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            None,
            precision=precision,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val

score_type = "hps_v2"
checkpoint_path = os.path.join("./pretrained_model/HPS_v2",'HPS_v2_compressed.pt')
initialize_model()
model = model_dict['model']
preprocess_val = model_dict['preprocess_val']
print(f'Loading model {score_type} ...')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
tokenizer = get_tokenizer(model_name)
model = model.to(device)
print(f'Loading model {score_type} successfully!')

def reverse_preprocessing(preprocessed_image, original_size=None):
    """
    Reverse the preprocessing steps applied to an image.

    Args:
        preprocessed_image (torch.Tensor): The preprocessed image tensor (shape [C, H, W]).
        original_size (tuple, optional): The original size of the image (H, W). Required for reversing resizing.

    Returns:
        PIL.Image: The image after reversing preprocessing.
    """
    # 1. Reverse normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)  # Normalization mean
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)   # Normalization std

    # Ensure the input is in the correct format
    if preprocessed_image.ndim == 4:  # If batch dimension exists, remove it
        preprocessed_image = preprocessed_image.squeeze(0)

    # Reverse normalization: x = (x * std) + mean
    denormalized_image = preprocessed_image * std[:, None, None] + mean[:, None, None]

    # 2. Clamp values to ensure they are in the valid range [0, 1]
    denormalized_image = denormalized_image.clamp(0, 1)

    # 3. Reverse resizing (optional, requires original size)
    if original_size:
        denormalized_image = torch.nn.functional.interpolate(
            denormalized_image.unsqueeze(0), size=original_size, mode="bilinear", align_corners=False
        ).squeeze(0)

    # 4. Convert back to PIL Image
    reversed_image = to_pil_image(denormalized_image)

    return reversed_image


# Assume 'model' is a pre-trained CLIP-like model
model.train()  # Set the model to evaluation mode

# Define the text prompt
prompt = ["'Product still of the new iPhone 2.0 in 2029.'"]
prompt = tokenizer(prompt[0]).to(device)

# Initialize the image as zeros
# Define the image size and mode
image_size = (224, 224)  # Width, height
image_mode = "RGB"  # Assuming a 3-channel RGB image

# Create a blank image with all values set to zero
raw_image = Image.open('./datasets/HPDv2_test/benchmark/benchmark_imgs/Deliberate/concept-art/00000.jpg')
image = preprocess_val(raw_image).unsqueeze(0).to(device)

noise = torch.randn_like(image).to(device) / 1000
optimized_image = (image + noise) # Remove batch dimension
outputs = model(image + noise, prompt)
image_features, text_features = outputs["image_features"], outputs["text_features"]
logits_per_image = image_features @ text_features.T * 100
diag = torch.diagonal(logits_per_image)
# Show the optimized image
plt.imshow(reverse_preprocessing(image[0], original_size=224))  # Convert tensor to PIL Image for display
plt.axis('off')
plt.title(f"Raw Image score {diag.mean()}")
plt.show()
plt.imshow(reverse_preprocessing(optimized_image, original_size=224))  # Convert tensor to PIL Image for display
plt.axis('off')
plt.title("Optimized Image")
plt.show()
# Show the optimized image
plt.imshow(reverse_preprocessing(noise, original_size=224))  # Convert tensor to PIL Image for display
plt.axis('off')
plt.title("noise Image")
plt.show()

noise.requires_grad = True
# Define optimizer
optimizer = torch.optim.Adam([noise], lr=0.0001)

# Optimization loop
for step in tqdm(range(1000)):  # Adjust the number of steps as needed
    optimizer.zero_grad()

    outputs = model(image+noise, prompt)
    image_features, text_features = outputs["image_features"], outputs["text_features"]
    logits_per_image = image_features @ text_features.T * 100
    diag = torch.diagonal(logits_per_image)

    # Define the loss (maximize logits for the text prompt)
    print(f"\nScore {diag.mean()}")
    print(f"Norm {torch.norm(noise, p='fro')}")
    loss = -diag.mean() / 3 + torch.norm(noise, p='fro')# Negative to maximize

    # Backpropagation
    loss.backward()
    optimizer.step()

    if step % 100 == 0:  # Print progress every 100 steps
        print(f"Step {step}, Loss: {loss.item()}")
        # After the optimization loop
        optimized_image = image + noise  # Remove batch dimension
        # Show the optimized image
        plt.imshow(reverse_preprocessing(optimized_image, original_size=224))  # Convert tensor to PIL Image for display
        plt.axis('off')
        plt.title(f"step {step} Optimized Image score {float(diag.mean())}")
        plt.show()
        # Reshape the image to merge H and W dimensions
        # Get the min and max for each channel independently
        plt.imshow(reverse_preprocessing(noise, original_size=224))  # Convert tensor to PIL Image for display
        plt.axis('off')
        plt.title(f"noise Image")
        plt.show()


# Compute the final score
with torch.no_grad():
    outputs = model(image+noise, prompt)
    image_features, text_features = outputs["image_features"], outputs["text_features"]
    logits_per_image = image_features @ text_features.T * 100
    diag = torch.diagonal(logits_per_image)
optimized_image = image + noise  # Remove batch dimension
# Show the optimized image
plt.imshow(reverse_preprocessing(optimized_image, original_size=224))  # Convert tensor to PIL Image for display
plt.axis('off')
plt.title(f"step final Optimized Image score {float(diag.mean())}")
plt.show()
print(f"Final Score for the prompt: {diag}")