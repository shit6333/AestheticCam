import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from aesthetics_model import AestheticsModel  
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

def load_and_preprocess_image(image_path, image_size=(224, 224), device='cuda'):
    """
    Load image from path, resize to (224, 224), convert to normalized tensor
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Converts to range [0,1]
    ])

    image = Image.open(image_path).convert("RGB")
    tensor_img = transform(image).unsqueeze(0).to(device)  # shape = [1, 3, 224, 224]
    return tensor_img

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AestheticsModel(device=device)

    image_files = [
        os.path.join(args.folder_path, fname)
        for fname in os.listdir(args.folder_path)
        if fname.lower().endswith(IMG_EXTENSIONS)
    ]
    if not image_files:
        print(f"No images found in folder: {args.folder_path}")
        return

    print(f"Evaluating {len(image_files)} images...\n")
    for img_path in sorted(image_files):
        try:
            image_tensor = load_and_preprocess_image(img_path, device=device)
            score = model(image_tensor)
            print(f"{os.path.basename(img_path):<30} Score: {score:.4f}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='/mnt/HDD3/miayan/omega/RL/gaussian-splatting/aes_test_images', help='Path to the image folder')
    args = parser.parse_args()
    main(args)
