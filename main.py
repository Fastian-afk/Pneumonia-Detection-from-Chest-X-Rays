from model import get_model
from transforms import get_transform
from utils import generate_gradcam_heatmaps
import os

# 👇 Set your validation directory path here
val_dir = "./val"  # Change to your dataset structure

# 📁 Create output folder
os.makedirs("gradcam_outputs", exist_ok=True)

# 🔧 Load model and transform
model = get_model()
transform = get_transform()

# 🚀 Generate Grad-CAM heatmaps
generate_gradcam_heatmaps(model, transform, val_dir, output_dir="gradcam_outputs", num_images=2)

print("✅ All Grad-CAM heatmaps saved to gradcam_outputs/")