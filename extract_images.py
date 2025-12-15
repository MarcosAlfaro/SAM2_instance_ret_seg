import webdataset as wds
import os
from PIL import Image
import io

# URLs
ILIAS_URLS = {
    'img_queries': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/ilias-core-queries-img-000000.tar",
    'core_db': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/ilias-core-db-000000.tar"
}

OUTPUT_DIR = "data/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_images(config_name):
    url = ILIAS_URLS[config_name]
    print(f"Extracting images for {config_name}...")
    
    # Decode to get jpg
    dataset = wds.WebDataset(url).decode("pil")
    
    count = 0
    for sample in dataset:
        key = sample['__key__']
        img = sample['jpg']
        
        # Save image
        # Key format: class/type/id. Replace / with _ to flatten
        safe_key = key.replace('/', '_')
        save_path = os.path.join(OUTPUT_DIR, f"{safe_key}.jpg")
        
        if not os.path.exists(save_path):
            img.save(save_path)
            
        count += 1
        if count % 1000 == 0:
            print(f"Extracted {count} images")

extract_images('img_queries')
extract_images('core_db')
