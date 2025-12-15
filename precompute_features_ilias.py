import torch
import numpy as np
import webdataset as wds
import torchvision.transforms as transforms
from PIL import Image
import os

# Configuración
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
# model.eval()

transform_dino = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# URLs based on the repository structure
ILIAS_URLS = {
    'img_queries': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/ilias-core-queries-img-000000.tar",
    'core_db': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/ilias-core-db-000000.tar",
    'distractors_5m': "https://huggingface.co/datasets/vrg-prague/ilias/resolve/main/mini_ilias_yfcc100m-{000000..000044}.tar"
}

def process_dataset(config_name, num_samples=None, extract_feats=False, extract_lbls=True):
    url = ILIAS_URLS[config_name]
    print(f"Processing {config_name} from {url}...")
    
    # Use webdataset to stream
    # We decode "pil" if we need features, otherwise we can just read keys?
    # To be safe and ensure exact same iteration order, we should use the same pipeline.
    # But decoding images is slow. 
    # If we only need labels, we can skip decoding.
    
    if extract_feats:
        dataset = wds.WebDataset(url).decode("pil").to_tuple("__key__", "jpg")
        # Load model only if needed
        global model
        if 'model' not in globals():
            print("Loading model...")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
            model.eval()
    else:
        # Just get keys
        dataset = wds.WebDataset(url).to_tuple("__key__")
    
    features_list = []
    labels_list = []
    
    count = 0
    for sample in dataset:
        if num_samples and count >= num_samples: break
        
        if extract_feats:
            key, img = sample
        else:
            key = sample[0]
            
        # Extract Label
        # Key format: class_name/type/id (e.g. bold_bimp_000/query/Q000_00)
        # For distractors, key is hash.
        if config_name == 'distractors_5m':
            label = "distractor"
        else:
            label = key.split('/')[0]
        
        if extract_lbls:
            labels_list.append(label)
            
        # Extract Features
        if extract_feats:
            # Ensure image is RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_tensor = transform_dino(img).unsqueeze(0).cuda()
            
            with torch.no_grad():
                emb = model(img_tensor)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            features_list.append(emb.cpu().numpy())
            
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} samples for {config_name}")
            
    # Save results
    if extract_lbls:
        lbls_arr = np.array(labels_list)
        np.save(f"{config_name}_labels.npy", lbls_arr)
        print(f"Saved {config_name}_labels.npy with shape {lbls_arr.shape}")
        
    if extract_feats:
        feats_arr = np.vstack(features_list)
        np.save(f"{config_name}_features_dinov2_small.npy", feats_arr) # Keeping name consistent
        print(f"Saved {config_name}_features_dinov2_small.npy with shape {feats_arr.shape}")

# 1. Process Queries
#process_dataset('img_queries', extract_feats=False, extract_lbls=True)

# 2. Process Core DB
#process_dataset('core_db', extract_feats=False, extract_lbls=True)

# 3. Process Distractors - Skipped as labels are not needed
process_dataset('distractors_5m', num_samples=1000000, extract_feats=False, extract_lbls=False)
