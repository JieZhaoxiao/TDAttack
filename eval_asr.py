import timm
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Load different pre-trained models based on name
def load_model(name):
    if name == "resnet":
        model = models.resnet50(pretrained=True)
    elif name == "vgg":
        model = models.vgg19(pretrained=True)
    elif name == "mobile":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "densenet":
        model = models.densenet121(pretrained=True)
    elif name == "vit-b":
        model = models.vit_b_16(pretrained=True)
    elif name == "swin-b":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "deit-b":
        model = timm.create_model('deit_base_patch16_224', pretrained=True)
    elif name == "cait-s":
        model = timm.create_model('cait_s24_224', pretrained=True)
    else:
        raise ValueError(f"Model {name} not recognized.")
    model.eval()
    return model

# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class labels from file
def load_labels(label_file):
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Load and preprocess multiple images as a batch
def load_batch_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)
    return torch.stack(images)

# Evaluate attack success rate across batches
def evaluate_attack(model, adv_images_folder, labels, num_images=1000, batch_size=200):
    attack_success_count = 0
    num_batches = num_images // batch_size

    for batch_idx in range(num_batches):
        adv_image_paths = [os.path.join(adv_images_folder, f"{i:04d}_adv_image.png")
                           for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)]

        adv_imgs = load_batch_images(adv_image_paths)

        orig_labels = [int(labels[i]) for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)]

        with torch.no_grad():
            adv_preds = model(adv_imgs).argmax(dim=1) + 1

        for orig_label, adv_pred in zip(orig_labels, adv_preds):
            if orig_label != adv_pred.item():
                attack_success_count += 1  # Check if prediction differs from original label (attack success)

        print(f"Processed batch {batch_idx+1}/{num_batches}...")

    return attack_success_count

# Main function to evaluate attack success across multiple models
def main():
    adv_images_folder = "output/resnet"
    label_file = "data/labels.txt"

    model_names = [
        "resnet",  "mobile", "vgg", "densenet", "vit-b", "swin-b", "deit-b", 'cait-s'
    ]
    labels = load_labels(label_file)

    with open("log.txt", "w") as log_file:
        log_file.write(" ".join(model_names) + "\n")

        success_rates = []
        for model_name in model_names:
            print(f"Evaluating attack success rate for {model_name}...")
            model = load_model(model_name)
            success_rate = evaluate_attack(model, adv_images_folder, labels, batch_size=200)

            success_rates.append(f"{success_rate*0.1}")
            print("success_rates:", success_rates)
        log_file.write(" ".join(success_rates) + "\n")

if __name__ == "__main__":
    main()