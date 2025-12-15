import webdataset as wds
import os
from tqdm import tqdm

def main():
    QUERY_TAR = "data/ilias-core-queries-img-000000.tar"
    OUTPUT_DIR = "data/images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Extracting queries from {QUERY_TAR} to {OUTPUT_DIR}...")
    
    dataset = wds.WebDataset(QUERY_TAR).decode("pil")
    
    count = 0
    for sample in tqdm(dataset):
        key = sample['__key__']
        img = sample['jpg']
        
        # Sanitize key as done in reranking script
        safe_key = key.replace('/', '_')
        save_path = os.path.join(OUTPUT_DIR, f"{safe_key}.jpg")
        
        img.save(save_path)
        count += 1
        
    print(f"Extracted {count} query images.")

if __name__ == "__main__":
    main()
