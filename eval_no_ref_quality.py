import torch
import argparse
import numpy as np
from PIL import Image
import os
from tqdm import trange
import pyiqa

# Define command-line arguments for image path, random seed, and image size
parser = argparse.ArgumentParser(description='Test Image Quality!')
parser.add_argument('--img_path', type=str, default='output/resnet')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--img_size', type=int, default=224)

args = parser.parse_args()
print(args)

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# Set device to GPU if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to load image files
def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return img
    except Exception as e:
        return None

# Main function to evaluate image quality using specified metrics
def evaluate_images_quality(img_dir, metrics_list, img_size=224, quant=False):
    # Initialize quality assessment metrics
    metrics = {}
    for metric_name in metrics_list:
        try:
            iqa_model = pyiqa.create_metric(metric_name, device=device)
            metrics[metric_name] = {
                'model': iqa_model,
                'lower_better': iqa_model.lower_better,
                'total_score': 0.0,
                'valid_count': 0
            }
            print(f" {metric_name} (lower_better={metrics[metric_name]['lower_better']})")
        except Exception as e:
            continue

    # Get list of adversarial image files
    img_files = [f for f in os.listdir(img_dir) if '_adv_image.png' in f]
    img_files.sort(key=lambda x: int(x.split('_')[0]))
    total_files = len(img_files)

    # Process each image file
    for i in trange(total_files):
        img_path = os.path.join(img_dir, img_files[i])
        pil_img = img_loader(img_path)

        if pil_img is None:
            continue

        # Resize image and convert to numpy array
        pil_img = pil_img.resize((img_size, img_size))
        img_np = np.array(pil_img).astype(np.float32)

        if quant:
            img_np = np.round(img_np)

        img_np = np.clip(img_np, 0, 255)

        # Convert numpy array to PyTorch tensor
        img_tensor = torch.from_numpy(img_np).to(device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

        # Calculate quality scores for each metric
        for metric_name in metrics_list:
            if metric_name not in metrics:
                continue
            try:
                score = metrics[metric_name]['model'](img_tensor)
                metrics[metric_name]['total_score'] += score.item()
                metrics[metric_name]['valid_count'] += 1
            except Exception as e:
                pass  # Skip any errors during metric calculation

    # Print final evaluation results
    print('*' * 60)
    for metric_name in metrics_list:
        if metric_name not in metrics:
            continue
        valid_count = metrics[metric_name]['valid_count']
        if valid_count > 0:
            avg_score = metrics[metric_name]['total_score'] / valid_count
            status = metrics[metric_name]['lower_better']
            print(f" {metric_name}:  {avg_score:.4f} ({status})")
    print('*' * 60)

# Main execution block
if __name__ == '__main__':
    evaluate_images_quality(
        img_dir=args.img_path,
        metrics_list=['hyperiqa', 'tres', 'musiq'],
        img_size=args.img_size,
        quant=False
    )