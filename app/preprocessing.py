from PIL import Image
import torchvision.transforms as transforms

# Image dimensions as used during training
IMG_WIDTH = 59
IMG_HEIGHT = 25

def preprocess_image(image: Image.Image) -> Image.Image:
    """Crop the image 3 pixels from the left and right as in training."""
    width, height = image.size
    return image.crop((3, 0, width - 3, height))

class Preprocess:
    def __call__(self, img):
        return preprocess_image(img)

def get_transform():
    """Return the transformation pipeline for the incoming image."""
    return transforms.Compose([
        Preprocess(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])
