from torchcam.methods import GradCAM
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch

def generate_gradcam_heatmaps(model, transform, val_dir, output_dir="gradcam_outputs", num_images=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam_extractor = GradCAM(model)

    classes = ["PNEUMONIA", "NORMAL"]
    for cls in classes:
        cls_path = os.path.join(val_dir, cls)
        images = os.listdir(cls_path)[:num_images]

        for i, img_name in enumerate(images):
            img_path = os.path.join(cls_path, img_name)
            image = Image.open(img_path)
            input_tensor = transform(image).unsqueeze(0).to(device)

            output = model(input_tensor)
            cam = cam_extractor(0, output)

            plt.imshow(image.convert("L"), cmap="gray")
            plt.imshow(cam[0].squeeze().cpu(), cmap="jet", alpha=0.5)
            plt.title(f"Grad-CAM - {cls} [{i}]")
            plt.axis("off")
            save_path = os.path.join(output_dir, f"{cls.lower()}_{i}.png")
            plt.savefig(save_path)
            plt.close()