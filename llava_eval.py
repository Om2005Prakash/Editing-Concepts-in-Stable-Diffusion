"""
accelerate launch llava_eval.py --dataset ./sample_dataset.csv --output_dir ./clip_edit
"""

import os
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from typing import List
from accelerate import Accelerator

from accelerate.utils import gather_object
from accelerate import PartialState
import argparse

# Initialize Accelerate
# accelerator = Accelerator()
# device = accelerator.device

BATCH_SIZE = 5  # adjust based on your GPU memory


def load_images_from_folder(folder: str) -> List[str]:
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def query_llava_batch(image_paths: List[str], prompt: str, device):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    queries = [f"<image>\nUSER: {prompt}\nASSISTANT:" for _ in images]
    inputs = processor(images=images, text=queries, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "pixel_values", "attention_mask"]}

    # adjust dims if needed
    if inputs["pixel_values"].dim() == 5:
        inputs["pixel_values"] = inputs["pixel_values"][:, 0]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    responses = processor.batch_decode(outputs, skip_special_tokens=True)
    return ["yes" if "yes" in resp.lower()[-5:] else "no" for resp in responses]


def evaluate_submission(model, submission_dir: str, challenge_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    # 🔹 initialize distributed state once
    distributed_state = PartialState()
    device = distributed_state.device
    model = model.to(device)

    concept_dirs = [d for d in os.listdir(submission_dir) if os.path.isdir(os.path.join(submission_dir, d))]
    pbar = tqdm(concept_dirs, desc="Evaluating concepts") if distributed_state.is_main_process else concept_dirs

    for concept_dir in pbar:
        if distributed_state.is_main_process:
            pbar.set_postfix_str(concept_dir)

        concept_path = os.path.join(submission_dir, concept_dir)
        subconcept_dirs = [d for d in os.listdir(concept_path) if os.path.isdir(os.path.join(concept_path, d))]

        for subconcept_dir in subconcept_dirs:
            subconcept_path = os.path.join(concept_path, subconcept_dir)
            row = challenge_df[challenge_df['concept_name'].str.replace('_', ' ') == subconcept_dir.replace('_', ' ')]
            if row.empty:
                continue
            info = row.iloc[0]
            image_paths = load_images_from_folder(subconcept_path)
            if not image_paths:
                continue

            # Build prompts
            direct_prompt = f"Is there a {info['concept_name']} in this picture? Answer with 'yes' or 'no'."
            prompt_variants = [('direct', direct_prompt)]

            partial = []
            data = [(cat, txt, img) for (cat, txt) in prompt_variants for img in image_paths]

            with distributed_state.split_between_processes(data) as proc_data:
                for (cat, txt, img) in proc_data:
                    resp = query_llava_batch([img], txt, device)
                    partial.append({
                        'image_id': os.path.basename(img),
                        'model_for_the_concept': concept_dir,
                        'checking_for_the_concept': subconcept_dir,
                        'prompt_type': cat,
                        'model_response': resp[0]
                    })

            # 🔹 gather results
            gathered = gather_object(partial)
            if distributed_state.is_main_process:
                print(len(gathered))
                results.extend(gathered)

    if distributed_state.is_main_process:
        df = pd.DataFrame(results)
        df['row_id'] = range(len(df))
        df.to_csv("evaluation_results.csv", index=False)
        print("Evaluation complete. Results saved to evaluation_results.csv")

parser = argparse.ArgumentParser(description="Evaluate LLaVA model on a challenge dataset.")
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to the challenge dataset CSV file."
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save submission results."
)
args = parser.parse_args()

# Load model
MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
dtype = torch.float16
model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=dtype)
processor = LlavaProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

if model.config.pad_token_id is None:
    model.config.pad_token_id = processor.tokenizer.pad_token_id
if model.config.eos_token_id is None:
    model.config.eos_token_id = processor.tokenizer.eos_token_id

# Load challenge dataset
challenge_df = pd.read_csv(args.challenge_df)

# Evaluate
df = evaluate_submission(model, args.submission_dir, challenge_df)
