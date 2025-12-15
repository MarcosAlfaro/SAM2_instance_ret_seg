import numpy as np
import webdataset as wds
import os
from PIL import Image
import io
import sys

# Configuration
RETRIEVAL_INDICES_PATH = "retrieval_indices.npy"
OUTPUT_DIR = "data/candidates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# URLs
ILIAS_URLS = {
    'img_queries': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/ilias-core-queries-img-000000.tar",
    'core_db': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/ilias-core-db-000000.tar",
    'distractors_100m': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/mini_ilias_yfcc100m-{000000..000044}.tar"
}

def main():
    if not os.path.exists(RETRIEVAL_INDICES_PATH):
        print(f"Error: {RETRIEVAL_INDICES_PATH} not found. Run ilias_retrieval.py first.")
        return

    print("Loading retrieval indices...")
    # Shape: (num_queries, top_k) -> e.g. (1232, 100)
    indices = np.load(RETRIEVAL_INDICES_PATH)
    
    # Flatten and get unique indices to avoid downloading duplicates
    unique_indices = np.unique(indices)
    print(f"Found {len(unique_indices)} unique images to extract.")
    
    # We need to know the offset for each dataset to map global index -> dataset specific index
    # Assuming the order in retrieval was: [core_db, distractors]
    # We need the lengths of each dataset.
    # Since we might not have the exact lengths loaded, we can try to infer or hardcode based on previous runs.
    # Core DB: 4715 images
    # Distractors: The rest
    
    CORE_DB_LEN = 4715
    
    # Split indices into Core DB and Distractors
    core_indices = unique_indices[unique_indices < CORE_DB_LEN]
    dist_indices = unique_indices[unique_indices >= CORE_DB_LEN]
    
    print(f"Core DB images to extract: {len(core_indices)}")
    print(f"Distractor images to extract: {len(dist_indices)}")
    
    # Create a set for fast lookup
    core_indices_set = set(core_indices)
    dist_indices_set = set(dist_indices)
    
    # 1. Extract Core DB Images
    if len(core_indices) > 0:
        extract_from_dataset('core_db', core_indices_set, offset=0)
        
    # 2. Extract Distractor Images
    if len(dist_indices) > 0:
        # Distractor indices start after Core DB
        # So global index 4715 corresponds to distractor index 0
        # We need to adjust the set for the function
        dist_indices_shifted = {i - CORE_DB_LEN for i in dist_indices}
        extract_from_dataset('distractors_100m', dist_indices_shifted, offset=CORE_DB_LEN)

def extract_from_dataset(config_name, target_indices_set, offset=0):
    url = ILIAS_URLS[config_name]
    print(f"Extracting from {config_name}...")
    
    # Use webdataset
    # We need to iterate and keep track of the index
    dataset = wds.WebDataset(url).decode("pil")
    
    count = 0
    extracted_count = 0
    
    try:
        for sample in dataset:
            # Current global index
            # For Core DB: count
            # For Distractors: count (relative)
            
            if count in target_indices_set:
                key = sample['__key__']
                img = sample['jpg']
                
                # Save with global index to easily find it later
                global_idx = offset + count
                save_path = os.path.join(OUTPUT_DIR, f"{global_idx}.jpg")
                
                if not os.path.exists(save_path):
                    img.save(save_path)
                
                extracted_count += 1
                if extracted_count % 100 == 0:
                    print(f"Extracted {extracted_count}/{len(target_indices_set)} images from {config_name}")
                
                # Optimization: If we found all, stop
                if extracted_count >= len(target_indices_set):
                    print(f"All target images extracted from {config_name}.")
                    break
            
            count += 1
            if count % 10000 == 0:
                print(f"Scanned {count} samples in {config_name}")
                
    except Exception as e:
        print(f"Error streaming {config_name}: {e}")
        print("Partial extraction completed. You can re-run to try and get the rest if the stream failed.")

if __name__ == "__main__":
    main()
