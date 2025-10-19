from dataclasses import dataclass

from tqdm import tqdm

import torch
from torch import autocast
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from einops import rearrange
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image

from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from diffusers import LMSDiscreteScheduler

from diffusers import StableDiffusionPipeline
import gc
import argparse
import random

import copy

parser = argparse.ArgumentParser(description="edit script")
parser.add_argument("edit_concept", type=str, help="")
parser.add_argument("target_concept", type=str, help="")
parser.add_argument("device", type=str, help="")

args = parser.parse_args()

@dataclass
class EditConfig:
    c = args.edit_concept           #Choose a concept from above list
    c_tar = args.target_concept
    descent_per_cycle = 6           #This is no. of decent step per data point to align with target concept
    learning_rate = 0.2             #Learning rate of Adam optimizer
    torch_device = args.device
    lam = 200                        #Weight complexity penality
    ts = [50, 10]                   #Timesteps that are used to denoise to align with target concepts
    num_inf = 10                    #Number of inference steps to generate image from pipeline
    batch_size = 8
    epochs = 10

config = EditConfig()

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

torch_device = config.torch_device

vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)

text_encoder_orig = pipe.text_encoder
text_encoder_orig = text_encoder_orig.to(torch_device)

for p in text_encoder_orig.parameters():
    p.requires_grad = False

for p in unet.parameters():
    p.requires_grad = False

for p in text_encoder.parameters():
    p.requires_grad = False

for p in vae.parameters():
    p.requires_grad = False

uncond_tokenized = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(torch_device)

uncond_embeddings = text_encoder(uncond_tokenized)["last_hidden_state"]

#Classifier Free Guidance
def generate(
        inf_timesteps:int,
        text_embeddings: torch.Tensor,
        guidance_scale = 7.5
):
    
    scheduler.set_timesteps(inf_timesteps)
    # text_embeddings = text_encoder(tokens)["last_hidden_state"]
    bs = text_embeddings.shape[0]
    text_embeddings = torch.cat([text_embeddings, uncond_embeddings.expand_as(text_embeddings)])

    latents = torch.randn(
        (bs, 4, 64, 64),
        device=torch_device
    )

    latents = latents * scheduler.init_noise_sigma

    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
            latent_model_input = torch.cat([latents]*2)
            # sigma = scheduler.sigmas[i]

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents

def clear_cache():
    torch.cuda.empty_cache()
    print(gc.collect())
    print(f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    print(f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

def plot_latents(latents):
    avg_latents = latents.mean(dim=1)

    grid = make_grid(avg_latents.unsqueeze(1), nrow=4, padding=2)
    img2 = ToPILImage()(torch.clamp((grid + 1)/2, 0, 1))
    return img2

layer_templete = {
        "inputs": [],
        "outputs": [],
        "orig_weight": None,
        "K_0K_0T": None,
        "K_pK_pT": None,
        "P": None,
        "in": None,
        "out": None,
        "bias": None,
        "name": None,
        "orig_weight": None,
        "curr_weight": None,
    }

layer_dict = {}

#Layers that needs to be edited
layers_to_modify = [
    # (text_encoder.text_model.encoder.layers[0].mlp.fc2, "text_encoder.text_model.encoder.layers[0].mlp.fc2"),
    (text_encoder.text_model.encoder.layers[1].mlp.fc2, "text_encoder.text_model.encoder.layers[1].mlp.fc2"),
    (text_encoder.text_model.encoder.layers[2].mlp.fc2, "text_encoder.text_model.encoder.layers[2].mlp.fc2"),
    (text_encoder.text_model.encoder.layers[3].mlp.fc2, "text_encoder.text_model.encoder.layers[3].mlp.fc2"),
    # (text_encoder.text_model.encoder.layers[4].mlp.fc2, "text_encoder.text_model.encoder.layers[4].mlp.fc2"),
    # (text_encoder.text_model.encoder.layers[5].mlp.fc2, "text_encoder.text_model.encoder.layers[5].mlp.fc2"),
    # (text_encoder.text_model.encoder.layers[6].mlp.fc2, "text_encoder.text_model.encoder.layers[6].mlp.fc2"),
    # (text_encoder.text_model.encoder.layers[7].mlp.fc2, "text_encoder.text_model.encoder.layers[7].mlp.fc2"),
    # (text_encoder.text_model.encoder.layers[8].mlp.fc2, "text_encoder.text_model.encoder.layers[8].mlp.fc2"),
    # (text_encoder.text_model.encoder.layers[9].mlp.fc2, "text_encoder.text_model.encoder.layers[9].mlp.fc2"),
    # (text_encoder.text_model.encoder.layers[10].mlp.fc2, "text_encoder.text_model.encoder.layers[10].mlp.fc2"),
    # (text_encoder.text_model.encoder.layers[11].mlp.fc2, "text_encoder.text_model.encoder.layers[11].mlp.fc2")
]

#Original layer weights
layers_orig = [
    # (text_encoder_orig.text_model.encoder.layers[0].mlp.fc2, "text_encoder.text_model.encoder.layers[0].mlp.fc2"),
    (text_encoder_orig.text_model.encoder.layers[1].mlp.fc2, "text_encoder.text_model.encoder.layers[1].mlp.fc2"),
    (text_encoder_orig.text_model.encoder.layers[2].mlp.fc2, "text_encoder.text_model.encoder.layers[2].mlp.fc2"),
    (text_encoder_orig.text_model.encoder.layers[3].mlp.fc2, "text_encoder.text_model.encoder.layers[3].mlp.fc2"),
    # (text_encoder_orig.text_model.encoder.layers[4].mlp.fc2, "text_encoder.text_model.encoder.layers[4].mlp.fc2"),
    # (text_encoder_orig.text_model.encoder.layers[5].mlp.fc2, "text_encoder.text_model.encoder.layers[5].mlp.fc2"),
    # (text_encoder_orig.text_model.encoder.layers[6].mlp.fc2, "text_encoder.text_model.encoder.layers[6].mlp.fc2"),
    # (text_encoder_orig.text_model.encoder.layers[7].mlp.fc2, "text_encoder.text_model.encoder.layers[7].mlp.fc2"),
    # (text_encoder_orig.text_model.encoder.layers[8].mlp.fc2, "text_encoder.text_model.encoder.layers[8].mlp.fc2"),
    # (text_encoder_orig.text_model.encoder.layers[9].mlp.fc2, "text_encoder.text_model.encoder.layers[9].mlp.fc2"),
    # (text_encoder_orig.text_model.encoder.layers[10].mlp.fc2, "text_encoder.text_model.encoder.layers[10].mlp.fc2"),
    # (text_encoder_orig.text_model.encoder.layers[11].mlp.fc2, "text_encoder.text_model.encoder.layers[11].mlp.fc2")
]

#For each layer, create a layer state which holds layer specific variable involved in update rule
for layer, name in layers_to_modify:
    t = copy.deepcopy(layer_templete)
    t["name"] = name + config.c
    t["in"] = layer.weight.shape[1]
    t["out"] = layer.weight.shape[0]
    t["orig_weight"] = layer.weight.detach().clone()

    try:
        t["bias"] = layer.bias.view(1, 1, -1)
    except:
        t["bias"] = torch.zeros((1, 1, t["out"]), device=torch_device)

    layer_dict[layer] = t

#Simply sample random timesteps and random noise and pass it through text model
#In this code we the text model has hooks and this function simply invokes those hooks

def pass_at_ts(
        dataloader: DataLoader,
        hook_handle,
):
    try:
        for batch in tqdm(dataloader):
            embbeds=text_encoder(batch.to(torch_device))["last_hidden_state"]

    finally:
        if hook_handle is not None:
            hook_handle.remove()

# Pass through text_model and unet with a given timestep and noise
# This function provides the signal for alignment
# Basically we call this fuction once with original prompt and again with edit prompt
# We use the update rule to align the internal representations

def pass_with_random_state(
        text_encoder,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        hook_handle = None,
):
    try:
        scheduler.set_timesteps(1000)
        latents = latents.to(torch_device)

        embbeds = text_encoder(tokens.to(torch_device))["last_hidden_state"]
        noise = noise.to(torch_device)

        timesteps = timesteps.to(torch_device)
        latents_t = scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(latents_t, timesteps, encoder_hidden_states=embbeds).sample

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return noise_pred

def latents_to_pil(latents):
    #Latent(B, 4, 64, 64) -> Image (3, 512, 512)
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def visualize_latents(out):
    for idx, group_latents in enumerate(out):
        images = latents_to_pil(group_latents.to(torch_device))  # Decode latents to list of PIL images

        fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
        fig.suptitle(f"Decoded images for out[{idx}]", fontsize=16)

        if len(images) == 1:
            axes = [axes]

        for img_idx, img in enumerate(images):
            axes[img_idx].imshow(img)
            axes[img_idx].axis('off')
            axes[img_idx].set_title(f"Image {img_idx}")

        plt.tight_layout()
        plt.show()

# Here we massage latents and corresponding tokens of OTHER concepts into a dataloader
# This is done as our update rule gurantees such tokens response will remain unchanged
# Then we calculate variables need for update rule, specifically null space (Here P)
token_dict = torch.load("token_dict")
not_c_tokens = torch.cat([token_dict[c_name]["concept"] for c_name in token_dict.keys() if c_name != config.c])

temp = []
for layer in layer_dict:
    def in_hook(module, input, output, layer=layer):
        layer_dict[layer]["inputs"].append(input[0].detach().cpu())
    
    temp.append(layer.register_forward_hook(in_hook))

_ = pass_at_ts(
        dataloader=[not_c_tokens],
        hook_handle=None,
    )

for layer in layer_dict:
    print()
    K_0 = torch.cat(layer_dict[layer]["inputs"], dim=0)
    K_0 = K_0.to(torch_device, dtype=torch.float32)
    K_0 = rearrange(K_0, "b p d -> (b p) d")

    print("K_0 Shape:", K_0.shape)
    print("K_0 Mean:", K_0.mean().item())

    K_0K_0T = K_0.T @ K_0
    layer_dict[layer]["inputs"].clear()

    # clear_cache()
    K_0K_0T = K_0K_0T.to(torch.float32)
    layer_dict[layer]["K_0K_0T"] = K_0K_0T
    layer_dict[layer]["K_pK_pT"] = K_0K_0T
    print("K_0K_0T Shape:",K_0K_0T.shape)
    print("K_0K_0T Mean:", K_0K_0T.mean().item())

    # clear_cache()
    U, S, _ = torch.linalg.svd(K_0K_0T, full_matrices=False)
    print("S mean:", torch.mean(S).item())

    threshold = torch.mean(S) / 90000  # Construct a small null space
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]

    print("Components below threshold:", small_singular_indices.shape[0])

    # Always select at least 100 smallest singular values
    if small_singular_indices.numel() < 100:
        # Get indices of 100 smallest singular values
        _, sorted_indices = torch.topk(S, k=100, largest=False)
        selected_indices = sorted_indices
    else:
        selected_indices = small_singular_indices

    print("Selected components:", len(selected_indices))

    P = U[:, selected_indices] @ U[:, selected_indices].T
    layer_dict[layer]["P"] = P

    del K_0
    del U, S

for layer in layer_dict:
    # K_0K_0T = torch.load(layer_dict[layer]["name"], map_location=torch_device)
    # P = torch.load(layer_dict[layer]["name"] + "P", map_location=torch_device)

    layer_dict[layer]["K_0K_0T"] = K_0K_0T
    layer_dict[layer]["K_pK_pT"] = K_0K_0T
    layer_dict[layer]["P"] = P

token_dict = torch.load("token_dict")

# c_tokenized = token_dict[config.c]["concept"]
# c_edit_tokenized = token_dict[config.c]["edit"]

def split_on_capitals(s: str) -> str:
    result = ""
    for char in s:
        if char.isupper() and result:  # avoid space before first letter
            result += " " + char
        else:
            result += char
    return result

c_tokenized = tokenizer(
    split_on_capitals(config.c),
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
).input_ids

c_edit_tokenized = tokenizer(
    split_on_capitals(config.c_tar),
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
).input_ids

class EmbbedsDataset(Dataset):
    def __init__(self, emb1, emb2) -> None:
        super().__init__()
        self.emb1 = emb1
        self.emb2 = emb2
    def __len__(self):
        return len(self.emb1)
    def __getitem__(self, index):
        return {
            "tokens": self.emb1[index],
            "tokens_tar": self.emb2[index],
        }

embbeds_per_concept = 8     #Sample with replacement from handcrafted tokens
edit_dataset = EmbbedsDataset(c_tokenized, c_edit_tokenized)
sampler = RandomSampler(edit_dataset, replacement=True, num_samples=embbeds_per_concept)
dataloader = DataLoader(edit_dataset, batch_size=config.batch_size, sampler=sampler)

for layer in layer_dict:
    del layer_dict[layer]["inputs"]
    del layer_dict[layer]["outputs"]
    layer_dict[layer]["inputs"] = []
    layer_dict[layer]["outputs"] = []

scaler = torch.GradScaler()

print(f"Editing {config.c}")
for e in range(config.epochs):
    print(f"############{e}############")
    for batch in dataloader:
        print(f"Getting Model Response (K_1)")

        # if torch.rand(1).item() < 0.5:
        print("orig")
        latents = generate(
            inf_timesteps=config.num_inf,
            text_embeddings = text_encoder_orig(batch["tokens"].to(torch_device))["last_hidden_state"]
        )

        for i, layer in enumerate(layer_dict):

            layer_dict[layer]["inputs"].clear()
            layer_dict[layer]["outputs"].clear()

            print(f"Editing Layer {i}")
            total_loss = 0.0

            def in_out_hook(module, input, output):
                layer_dict[layer]["inputs"].append(input[0].detach())
                bias = layer_dict[layer]["bias"].detach()
                layer_dict[layer]["outputs"].append((output - bias).detach())

            hook_handle = layer.register_forward_hook(in_out_hook)
            noise = torch.randn_like(latents, device=torch_device)
            bs = noise.shape[0]
            timesteps = torch.randint(config.ts[1], config.ts[0]+1, (bs,), device=torch_device).long()

            with autocast("cuda"):
                _ = pass_with_random_state(
                    text_encoder=text_encoder,
                    latents=latents,
                    tokens=batch["tokens"],
                    noise=noise,
                    timesteps=timesteps,
                    hook_handle=hook_handle,
                )

            inputs_new, outputs_new = layer_dict[layer]["inputs"], layer_dict[layer]["outputs"]
            inputs_new, outputs_new = torch.cat(inputs_new, dim=0), torch.cat(outputs_new, dim=0)

            z = (outputs_new.detach()).clone().requires_grad_()

            layer_dict[layer]["outputs"].clear()

            optimizer = torch.optim.Adam([z], lr=config.learning_rate)
            bias = layer_dict[layer]["bias"].detach()
            
            for step in range(config.descent_per_cycle):

                def mod_hook(module, input, output):
                    return z + bias
                
                hook_handle = layer.register_forward_hook(mod_hook)

                with autocast("cuda"):
                    noise_pred = pass_with_random_state(
                        text_encoder=text_encoder_orig,
                        latents=latents,
                        tokens=batch["tokens"],
                        noise=noise,
                        timesteps=timesteps,
                        hook_handle=hook_handle,
                    )

                with autocast("cuda"):
                    noise_tar = pass_with_random_state(
                        text_encoder=text_encoder,
                        latents=latents,
                        tokens=batch["tokens_tar"],
                        noise=noise,
                        timesteps=timesteps,
                        hook_handle=hook_handle,
                    )
                
                loss = F.mse_loss(noise_pred, noise_tar)

                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.update()
                print(f"Epoch {e} Step {step} Loss: {loss.item():.6f}")

            W = layer.weight

            P = layer_dict[layer]["P"].to(torch.float32)

            K_1 = rearrange(inputs_new, "b p d -> (b p) d")
            K_1 = K_1.to(torch.float32)
            K_1K_1T = K_1.T @ K_1

            z = rearrange(z, "b p d -> (b p) d")
            z = z.to(torch.float32)
            V_1K_1T = z.T @ K_1 

            RK_1T = V_1K_1T - W @ K_1K_1T

            A = (K_1K_1T @ P + config.lam * torch.eye(K_1K_1T.shape[0], device=torch_device)).T
            B = (RK_1T @ P).T

            upd_matrix = torch.linalg.solve(
                A, B
            ).T
            print(upd_matrix.mean())
            
            layer.weight.data.copy_(layer.weight + upd_matrix)
            layer_dict[layer]["K_pK_pT"] = K_1K_1T

            layer_dict[layer]["inputs"].clear()
            layer_dict[layer]["outputs"].clear()

            print(f"Layer_{i} ‖upd_norm‖: {torch.norm(upd_matrix).item():.4f}")

    # visualize_latents([generate(inf_timesteps=config.num_inf, text_embeddings=text_encoder(batch["tokens"][:4].to(torch_device))["last_hidden_state"])])
    # # visualize_latents([latents[:4]])
    # visualize_latents([generate(inf_timesteps=config.num_inf, text_embeddings=text_encoder(prompt_tokenized1.to(torch_device))["last_hidden_state"])])
    # visualize_latents([generate(inf_timesteps=config.num_inf, text_embeddings=text_encoder(prompt_tokenized2.to(torch_device))["last_hidden_state"])])

from pathlib import Path
import io
import zipfile

path = Path(f"./{config.c}")
path.mkdir(parents=True, exist_ok=True)

for layer in layer_dict:
    torch.save(layer, path / f"{layer_dict[layer]['name']}")