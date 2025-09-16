import torch
import gradio as gr
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh
from PIL import Image

# pick device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models once
print("Loading Shap-E models (this may take a bit)...")
xm = load_model('transmitter', device=device)
im_model = load_model('image300M', device=device)

def generate_3d(image):
    """Takes an uploaded image and returns path to generated 3D model"""
    img = image.convert("RGB")

    # generate latents
    latents = sample_latents(
        batch_size=1,
        model=im_model,
        guidance_scale=3.0,
        model_kwargs=dict(images=[img]),
        device=device
    )

    # decode into mesh
    mesh = decode_latent_mesh(xm, latents[0])

    # save model
    output_path = "output.obj"
    with open(output_path, "w") as f:
        mesh.write_obj(f)

    return output_path

# Gradio UI
demo = gr.Interface(
    fn=generate_3d,
    inputs=gr.Image(type="pil"),
    outputs=gr.File(),
    title="Shap-E: 2D → 3D Model",
    description="Upload a 2D image and download a generated 3D model (.obj)"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
