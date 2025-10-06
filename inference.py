"""
Example Usage:

python generation.py concept_name prompt /path/to/output --num_images 10 --batch_size 10

For Example Van Gogh
python example.py VanGogh "a painting of a cat in Van Gogh style" ./out/VanGogh --num_images 10 --batch_size 10
"""

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from typing import List, Tuple
from tqdm.auto import tqdm

# Set the device to GPU
torch_device = torch.device("cuda:0")


def load_model(concept: str) -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion model from `concept` model identifier.
    """
    # concept = concept.replace(" ", "")  # Remove spaces for clean folder names
    base_path = Path(f"./{concept}")
    pipe = StableDiffusionPipeline.from_pretrained(base_path / concept)
    return pipe


def run_inference(pipe: StableDiffusionPipeline, prompt_path: List[Tuple[str, Path]], batch_size: int):
    """
    Run inference on batches of prompts and save generated images to corresponding output directories.
    """
    pipe = pipe.to(torch_device)
    for batch_start in tqdm(range(0, len(prompt_path), batch_size), desc="Generating"):
        batch = prompt_path[batch_start: batch_start + batch_size]
        prompts, paths = zip(*batch)

        # Generate images
        prompts = list(prompts)
        images = pipe(prompts).images

        # Save each image to its corresponding path
        for i, (image, path) in enumerate(zip(images, paths)):
            image.save(path / f"idx_{batch_start + i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a text prompt using a concept-specific Stable Diffusion model."
    )

    parser.add_argument("concept_name", type=str,
                        help="Model name or local path to the concept-specific Stable Diffusion model (e.g., './Dancing').")
    parser.add_argument("prompt", type=str,
                        help="The text prompt to use for image generation.")
    parser.add_argument("output_path", type=str,
                        help="Directory where generated images will be saved.")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images to generate for the prompt.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for image generation.")

    args = parser.parse_args()

    # Load model and prepare paths
    prompt = args.prompt
    print(prompt)
    pipe = load_model(args.concept_name)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    IMGS = args.num_images
    BATCH_SIZE = args.batch_size

    # Create list of (prompt, output_path) pairs
    prompt_path = [(prompt, output_path)] * IMGS

    # Run generation
    run_inference(pipe, prompt_path, BATCH_SIZE)