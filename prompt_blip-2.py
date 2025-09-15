"""
Generate image descriptions for a folder of PNGs using BLIP-2 and save them to a text file.
"""

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os

# -----------------------------------------------------------------------------
# Model & processor initialization
# -----------------------------------------------------------------------------
# Load the BLIP-2 processor for preprocessing images and decoding model outputs.
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Load the BLIP-2 captioning model.
# device_map="auto" will place model shards on available devices (e.g., GPU) if present.
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

# -----------------------------------------------------------------------------
# I/O configuration
# -----------------------------------------------------------------------------
# Directory containing input images and the path to the output descriptions file.
image_dir = "dataset/images"
output_file = "adversarial_prompt.txt"

# -----------------------------------------------------------------------------
# Discover and sort input images
# -----------------------------------------------------------------------------
# Collect *.png files and sort them numerically by filename (e.g., "1.png", "2.png", ...).
# Assumes filenames before the extension are integers.
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith('.png')],
    key=lambda x: int(x.split('.')[0])
)

# -----------------------------------------------------------------------------
# Caption generation loop
# -----------------------------------------------------------------------------
# Open the output file once and stream descriptions as they are generated.
with open(output_file, 'w') as f:
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = processor(image, return_tensors="pt").to("cuda")
            generated_ids = model.generate(**inputs, max_length=50)
            description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            f.write(f"{description}")
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
