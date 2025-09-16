# Shap-E 2D → 3D in Codespaces

This repo turns a 2D image into a downloadable 3D model using [Shap-E](https://github.com/openai/shap-e).

## How to Use
1. Open this repo in GitHub Codespaces.
2. Upload your image (rename it `input.png`).
3. Run:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
4. Download `output.obj` from the file explorer.

You now have a 3D model you can open in Blender, Unity, etc.
