"""
This script performs multi GPU inference on edited model.
Basically it builds a SD pipeline and then loads the edited weight and use accelerate to distribute the inference


Example Usage:

First config accelerate with:
```bash
accelerate config default
```

then run the script with:
accelerate launch --num_processes generation.py /path/to/dataset.csv /path/to/output --imgs 10 --batch_size 10

Example:
accelerate launch generation.py ./challenge_dataset.csv ./clip_edit/ --imgs 3 --batch_size 16
"""

import argparse
from pathlib import Path
import pandas as pd

from accelerate import PartialState
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline
from typing import List, Tuple
from tqdm.auto import tqdm
from diffusers import LMSDiscreteScheduler

def run_inference(pipe: StableDiffusionPipeline, prompt_path: List[Tuple[str, Path]], batch_size: int):
    distributed_state = PartialState()

    pipe.to(distributed_state.device)
    with distributed_state.split_between_processes(prompt_path) as prompts_path_for_this_process:
        
        for batch_start in range(0, len(prompts_path_for_this_process), batch_size):
            batch = prompts_path_for_this_process[batch_start: batch_start + batch_size]

            prompts, paths, prompt_type = zip(*batch)
            
            prompts = list(prompts)
            
            with autocast("cuda"):
                images = pipe(prompts, num_inference_steps=10).images

            for i, (image, path, prompt_type) in enumerate(zip(images, paths, prompt_type)):
                image.save(path / f"image_rank_{distributed_state.process_index}_idx_{batch_start + i} {prompt_type}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gen script")

    parser.add_argument("dataset_path", type=str, help="")
    parser.add_argument("output_path", type=str, help="")
    parser.add_argument("--imgs_prompt", type=int, default=10,
                    help="Images per concept prompt")
    parser.add_argument("--batch_size", type=int, default=10,
                    help="")

    args = parser.parse_args()
    challenge_df = pd.read_csv(args.dataset_path)
    output_path = Path(args.output_path)
    IMG_EACH_CONCEPT_PROMPT = args.imgs_prompt
    BATCH_SIZE = args.batch_size

    # assert set(pipe_dict.keys()) == set(challenge_df["concept_name"]), \
    # "Mismatch between model names and concept names"

    ##Prepare prompts and directory structure
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe.safety_checker = None
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    pipe.scheduler = scheduler
    
    for c in ["LabradorRetriever", "BarbetonDaisy", "BlueJay", "GolfBall", "AppleFruit", "VanGough", "Doodle", "Neon", "Monet", "Sketch", "Wedding", "Sunset", "Rainfall", "AuroraBorialis", "Scenery", "Sleeping", "Walking", "Eating", "Dancing", "Jumping"]:
    # for c in ["LabradorRetriever"]:
        prompt_path = []
        # c = row["concept_name"]
        print(c)
        for _, sc_row in challenge_df.iterrows():
            sc = sc_row["concept_name"]
            output_dir = output_path / c / sc
            output_dir.mkdir(exist_ok=True, parents=True)
            prompt_path.extend(
                [
                    (sc_row["direct_prompt_1"], output_dir, "direct"),
                    (sc_row["direct_prompt_2"], output_dir, "direct"),
                    (sc_row["direct_prompt_3"], output_dir, "direct"),
                    (sc_row["direct_prompt_4"], output_dir, "direct"),
                    (sc_row["direct_prompt_5"], output_dir, "direct"),
                    (sc_row["indirect_prompt_1"], output_dir, "indirect"),
                    (sc_row["indirect_prompt_2"], output_dir, "indirect"),
                    (sc_row["indirect_prompt_3"], output_dir, "indirect"),
                    (sc_row["indirect_prompt_4"], output_dir, "indirect"),
                    (sc_row["indirect_prompt_5"], output_dir, "indirect"),
                    # (sc_row["adversarial_prompt"], output_dir, "adversarial"),
                    # (sc_row["adversarial_prompt"], output_dir, "adversarial"),
                    # (sc_row["adversarial_prompt"], output_dir, "adversarial"),
                    # (sc_row["adversarial_prompt"], output_dir, "adversarial"),
                    # (sc_row["adversarial_prompt"], output_dir, "adversarial"),
                ] * args.imgs_prompt
            )
        
        text_encoder = pipe.text_encoder
        layers = [
            (text_encoder.text_model.encoder.layers[1].mlp.fc2, "text_encoder.text_model.encoder.layers[1].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[2].mlp.fc2, "text_encoder.text_model.encoder.layers[2].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[3].mlp.fc2, "text_encoder.text_model.encoder.layers[3].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[4].mlp.fc2, "text_encoder.text_model.encoder.layers[4].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[5].mlp.fc2, "text_encoder.text_model.encoder.layers[5].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[6].mlp.fc2, "text_encoder.text_model.encoder.layers[6].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[7].mlp.fc2, "text_encoder.text_model.encoder.layers[7].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[8].mlp.fc2, "text_encoder.text_model.encoder.layers[8].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[9].mlp.fc2, "text_encoder.text_model.encoder.layers[9].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[10].mlp.fc2, "text_encoder.text_model.encoder.layers[10].mlp.fc2"),
            (text_encoder.text_model.encoder.layers[11].mlp.fc2, "text_encoder.text_model.encoder.layers[11].mlp.fc2")
            ]
    
        for layer in layers:
            weight = torch.load(f"./{c}/{layer[1]}"+c, weights_only=False).weight
            layer[0].weight.data.copy_(weight)
    
        run_inference(pipe, prompt_path, BATCH_SIZE)