import torch
import numpy as np
import os
from PIL import Image
import argparse
from tqdm import tqdm
import time
import json

# Import SANSA model components
from models.sansa.sansa_original import build_sansa
from util.promptable_utils import build_prompt_dict
import torchvision.transforms as transforms
import torch.nn.functional as F

def main():
    # Configuration
    RETRIEVAL_INDICES_PATH = "npy_files/retrieval_indices.npy"
    CANDIDATES_DIR = "data/candidates"
    QUERY_IMAGES_DIR = "data/images"
    METADATA_PATH = "npy_files/img_queries_bboxes.npy"
    QUERY_KEYS_PATH = "npy_files/img_queries_keys.npy"
    CHECKPOINT_PATH = "checkpoints/sansa_revisitop_epoch10.pt"  # Load trained checkpoint
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Retrieval Results
    print("Loading retrieval results...")
    try:
        retrieval_indices = np.load(RETRIEVAL_INDICES_PATH)
        query_bboxes = np.load(METADATA_PATH)
        query_keys = np.load(QUERY_KEYS_PATH)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    num_queries, top_k = retrieval_indices.shape
    print(f"Processing {num_queries} queries with {top_k} candidates each.")

    # Load Model
    print("Loading SAM2 model...")
    sam2_version = "tiny"
    adaptformer_stages = []
    channel_factor = 4
    
    model = build_sansa(sam2_version, adaptformer_stages, channel_factor, DEVICE, obj_pred_scores=True)
    
    # Load trained checkpoint
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    model.to(DEVICE)
    model.eval()
    
    # Transforms
    img_size = 518
    
    def custom_to_tensor(pic):
        img_np = np.array(pic)
        img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
        return img / 255.0

    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.Lambda(custom_to_tensor),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Re-ranking Loop
    reranked_indices = []
    reranked_scores = []
    
    for q_idx in tqdm(range(num_queries), desc="Testing trained model"):
        # 1. Load Query Image & Prompt
        q_key = query_keys[q_idx]
        safe_q_key = q_key.replace('/', '_')
        q_img_path = os.path.join(QUERY_IMAGES_DIR, f"{safe_q_key}.jpg")
        
        if not os.path.exists(q_img_path):
            print(f"Warning: Query image {q_img_path} not found. Skipping.")
            reranked_indices.append(retrieval_indices[q_idx])
            reranked_scores.append(np.zeros(top_k))
            continue
            
        query_img_pil = Image.open(q_img_path).convert('RGB')
        query_img = transform(query_img_pil).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Prepare Prompt (BBox)
        bbox = query_bboxes[q_idx]
        x, y, w, h = bbox
        
        # Scale bbox to img_size
        orig_w, orig_h = query_img_pil.size
        scale_x = img_size / orig_w
        scale_y = img_size / orig_h
        
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, img_size - 1))
        y1 = max(0, min(y1, img_size - 1))
        x2 = max(x1 + 1, min(x2, img_size))
        y2 = max(y1 + 1, min(y2, img_size))
        
        # Build prompt dict manually with box coordinates
        point_coords = torch.tensor([[[x1, y1], [x2, y2]]], dtype=torch.float32, device=DEVICE)
        point_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=DEVICE)
        
        prompt_dict = {
            'shots': 1,
            0: {
                0: {
                    'prompt_type': 'box',
                    'prompt': {
                        'point_coords': point_coords,
                        'point_labels': point_labels
                    }
                }
            }
        }
        
        # 2. Process Candidates
        candidate_indices = retrieval_indices[q_idx]
        scores = []
        
        # Batch processing for candidates
        BATCH_SIZE = 16
        
        for i in range(0, top_k, BATCH_SIZE):
            batch_indices = candidate_indices[i:i+BATCH_SIZE]
            
            # Load candidate images
            batch_imgs = []
            valid_batch_indices = []
            
            for cand_idx in batch_indices:
                cand_path = os.path.join(CANDIDATES_DIR, f"{cand_idx}.jpg")
                if os.path.exists(cand_path):
                    try:
                        img = Image.open(cand_path).convert('RGB')
                        img_t = transform(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        batch_imgs.append(img_t)
                        valid_batch_indices.append(True)
                    except Exception:
                        valid_batch_indices.append(False)
                else:
                    valid_batch_indices.append(False)
            
            if not batch_imgs:
                scores.extend([0.0] * len(batch_indices))
                continue
                
            # Concatenate: [Query, Cand1, Cand2, ...]
            imgs_tensor = torch.cat([query_img] + batch_imgs, dim=1).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(imgs_tensor, prompt_dict)
                object_score_logits = outputs["object_score_logits"]
                
            # Extract scores
            batch_scores = []
            logit_idx = 1
            for is_valid in valid_batch_indices:
                if is_valid:
                    if logit_idx < len(object_score_logits):
                        score = torch.sigmoid(object_score_logits[logit_idx]).item()
                        batch_scores.append(score)
                        logit_idx += 1
                    else:
                        batch_scores.append(0.0)
                else:
                    batch_scores.append(0.0)
            
            scores.extend(batch_scores)
            
        # 3. Re-rank
        scores = np.array(scores)
        sorted_arg_indices = np.argsort(scores)[::-1]
        
        reranked_cand_indices = candidate_indices[sorted_arg_indices]
        reranked_cand_scores = scores[sorted_arg_indices]
        
        reranked_indices.append(reranked_cand_indices)
        reranked_scores.append(reranked_cand_scores)

        # Intermediate logging
        if q_idx % 50 == 0:
            print(f"\n[Query {q_idx}] Processed. Valid candidates: {sum(1 for s in scores if s > 0)}")
            #print(f"  Top 5 Scores: {reranked_cand_scores[:5]}")
            print(f"  Top 10 Indices: {reranked_cand_indices[:10]}")
        
    # Save Results
    print("Saving re-ranking results...")
    np.save("npy_files/reranked_indices_trained.npy", np.array(reranked_indices))
    np.save("npy_files/reranked_scores_trained.npy", np.array(reranked_scores))
    
    # Calculate mAP and Recall
    print("Calculating metrics...")
    try:
        # Load labels
        q_labels = np.load("npy_files/img_queries_labels.npy")
        db_labels = np.load("npy_files/core_db_labels.npy")
        
        num_distractors = 1000000
        dist_labels = np.full(num_distractors, -1, dtype=db_labels.dtype)
        database_labels = np.concatenate([db_labels, dist_labels])
        
        # Calculate mAP@k
        k_values = [1, 5, 10, 20, 50, 100]
        
        aps = {k: [] for k in k_values}
        recalls = {k: [] for k in k_values}
        
        reranked_indices = np.array(reranked_indices)
        
        for i in range(num_queries):
            q_lbl = q_labels[i]
            retrieved_indices = reranked_indices[i]
            retrieved_lbls = database_labels[retrieved_indices]
            
            relevant = (retrieved_lbls == q_lbl).astype(int)
            total_relevant = np.sum(db_labels == q_lbl)
            
            if total_relevant == 0:
                continue
                
            for k in k_values:
                if k > len(relevant):
                    rel_k = relevant
                else:
                    rel_k = relevant[:k]
                
                score = 0.0
                num_hits = 0.0
                
                for j in range(len(rel_k)):
                    if rel_k[j] == 1:
                        num_hits += 1
                        score += num_hits / (j + 1)
                
                ap = score / min(total_relevant, k)
                aps[k].append(ap)
                
                recall = np.sum(rel_k) / total_relevant
                recalls[k].append(recall)
        
        print("\n=== TRAINED MODEL Re-ranking Results ===")
        for k in k_values:
            map_k = np.mean(aps[k])
            mean_recall_k = np.mean(recalls[k])
            print(f"mAP@{k}: {map_k:.4f} | R@{k}: {mean_recall_k:.4f}")
        
        # Compare with baseline
        print("\n=== Comparison with Baseline ===")
        print("Baseline DINOv2 results:")
        print("mAP@1: 0.3515 | mAP@5: 0.2460 | mAP@10: 0.2045 | mAP@20: 0.1773 | mAP@50: 0.1572 | mAP@100: 0.1479")
        print("\nUntrained SAM2 results:")
        print("mAP@1: 0.0966 | mAP@5: 0.0718 | mAP@10: 0.0643 | mAP@20: 0.0597 | mAP@50: 0.0550 | mAP@100: 0.0509")
        print(f"\nTrained SAM2 (epoch {checkpoint['epoch']}) results:")
        for k in k_values:
            print(f"mAP@{k}: {np.mean(aps[k]):.4f}", end=" | ")
        print()
            
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting complete!")

if __name__ == "__main__":
    main()
