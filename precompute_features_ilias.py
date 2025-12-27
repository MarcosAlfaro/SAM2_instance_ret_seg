import torch
import numpy as np
import webdataset as wds
import torchvision.transforms as transforms
from PIL import Image
import os
from util import load_model, image_transform

# URLs based on the repository structure
ILIAS_URLS = {
    'img_queries': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/ilias-core-queries-img-000000.tar",
    'core_db': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/ilias-core-db-000000.tar",
    'distractors_5m': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/mini_ilias_yfcc100m-{000000..000044}.tar"
}

def get_last_processed_shard(model_name, config_name):
    """Get the last processed shard number from checkpoint file."""
    checkpoint_file = f"npy_files/{model_name}/{config_name}_last_shard.txt"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return -1

def save_last_processed_shard(model_name, config_name, shard_num):
    """Save the last processed shard number to checkpoint file."""
    checkpoint_file = f"npy_files/{model_name}/{config_name}_last_shard.txt"
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        f.write(str(shard_num))

def extract_features(model, processor, img, model_name):
    """Extract features from image using the specified model."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Handle different model types
    if model_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        # DINOv2: Use manual transform
        tf = image_transform.get_transform(model_name=model_name)
        img_tensor = tf(img).unsqueeze(0).cuda()
        
        with torch.no_grad():
            emb = model(img_tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()
    
    elif model_name == 'siglip-base-patch16-224':
        # SigLIP: Use processor for image processing
        inputs = processor(images=img, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            emb = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return emb.cpu().numpy()
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def process_dataset(config_name, num_samples=None, model_name=None, extract_feats=False, extract_lbls=True):
    url = ILIAS_URLS[config_name]
    print(f"Processing {config_name} from {url}")
    
    # Create output directory
    output_dir = f"npy_files/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we're resuming from a checkpoint (only for distractors with shards)
    start_shard = 0
    if config_name == 'distractors_5m' and extract_feats:
        start_shard = get_last_processed_shard(model_name, config_name) + 1
        if start_shard > 0:
            print(f"Resuming from shard {start_shard}")
    
    # Use webdataset to stream
    if extract_feats:
        dataset = wds.WebDataset(url).decode("pil").to_tuple("__key__", "jpg")
    else:
        dataset = wds.WebDataset(url).to_tuple("__key__")
    
    features_list, labels_list = [], []

    # Load model and transform
    model, processor = load_model(model_name=model_name, device='cuda')
    model.eval()  # Set to evaluation mode

    count = 0
    current_shard = -1
    
    for sample in dataset:
        if num_samples and count >= num_samples: break
        
        if extract_feats:
            key, img = sample
        else:
            key = sample[0]
        
        # Track shard number for distractors
        if config_name == 'distractors_5m':
            # Extract shard number from key or track based on URL pattern
            # Assuming webdataset provides shard info or we track by count
            new_shard = count // 100000  # Approximate shard size
            if new_shard != current_shard and current_shard >= 0 and extract_feats:
                # Save features for completed shard
                if len(features_list) > 0:
                    feats_arr = np.vstack(features_list)
                    shard_file =  f"{output_dir}/{config_name}_features_shard_{current_shard:06d}.npy"
                    np.save(shard_file, feats_arr)
                    print(f"Saved {shard_file} with shape {feats_arr.shape}")
                    features_list = []  # Clear for next shard
                    
                    # Save checkpoint
                    save_last_processed_shard(model_name, config_name, current_shard)
            
            current_shard = new_shard
            
            # Skip if we already processed this shard
            if current_shard < start_shard:
                count += 1
                continue
            
        # Extract Label
        # Key format: class_name/type/id (e.g. bold_bimp_000/query/Q000_00)
        # For distractors, key is just a number
        label = "distractor" if config_name == 'distractors_5m' else key.split('/')[0]
        if extract_lbls:
            labels_list.append(label)
            
        # Extract Features
        if extract_feats:
            emb = extract_features(model, processor, img, model_name)
            features_list.append(emb)
            
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} samples for {config_name}")
    
    # Save final results
    if extract_lbls:
        lbls_arr = np.array(labels_list)
        labels_file =  f"{output_dir}/{config_name}_labels.npy"
        np.save(labels_file, lbls_arr)
        print(f"Saved {labels_file} with shape {lbls_arr.shape}")
        
    if extract_feats:
        if config_name == 'distractors_5m':
            # Save remaining features for last shard
            if len(features_list) > 0:
                feats_arr = np.vstack(features_list)
                shard_file =  f"{output_dir}/{config_name}_features_shard_{current_shard:06d}.npy"
                np.save(shard_file, feats_arr)
                print(f"Saved final {shard_file} with shape {feats_arr.shape}")
                
                # Save final checkpoint
                save_last_processed_shard(model_name, config_name, current_shard)
        else:
            # For the other sets, save all at once
            feats_arr = np.vstack(features_list)
            features_file = output_dir / f"{config_name}_features.npy"
            np.save(features_file, feats_arr)
            print(f"Saved {features_file} with shape {feats_arr.shape}")

# 1. Process Queries
process_dataset('img_queries', model_name='siglip-base-patch16-224', extract_feats=True, extract_lbls=True)

# 2. Process Core DB
process_dataset('core_db', model_name='siglip-base-patch16-224', extract_feats=True, extract_lbls=True)

# 3. Process Distractors - Skipped as labels are not needed
process_dataset('distractors_5m', num_samples=5000000, model_name='siglip-base-patch16-224', extract_feats=True, extract_lbls=False)