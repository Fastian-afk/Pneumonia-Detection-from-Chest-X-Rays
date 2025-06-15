from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale

def get_transform():
    return Compose([
        Resize((224, 224)),
        Grayscale(num_output_channels=3),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])