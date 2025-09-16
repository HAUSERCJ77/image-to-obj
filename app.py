import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh
from PIL import Image

# pick device (Codespaces is usually CPU, but if GPU is enabled it'll use that)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
print("Loading models...")
xm = load_model('transmitter', device=device)
im_model = load_model('image300M', device=device)

# load your input image
img_path = "input.png"  # change if needed
print(f"Loading image: {img_path}")
img = Image.open(img_path).convert("RGB")

# generate latents
print("Generating 3D latent...")
latents = sample_latents(
    batch_size=1,
    model=im_model,
    guidance_scale=3.0,
    model_kwargs=dict(images=[img]),
    device=device
)

# decode into mesh
print("Decoding mesh...")
mesh = decode_latent_mesh(xm, latents[0])

# save as OBJ
output_path = "output.obj"
with open(output_path, "w") as f:
    mesh.write_obj(f)

print(f"✅ Done! 3D model saved to {output_path}")
