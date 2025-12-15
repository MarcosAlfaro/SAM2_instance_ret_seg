import torch
import numpy as np
import os
from PIL import Image
import argparse
from tqdm import tqdm
import time
import json

# Import SANSA model components
# Assuming the user's workspace structure
from models.sansa.sansa_original import build_sansa
from util.promptable_utils import build_prompt_dict
import torchvision.transforms as transforms
import torch.nn.functional as F

def main():
    # Configuration
    RETRIEVAL_INDICES_PATH = "retrieval_indices.npy"
    CANDIDATES_DIR = "data/candidates"
    QUERY_IMAGES_DIR = "data/images" # Assuming queries were extracted here
    METADATA_PATH = "img_queries_bboxes.npy" # Need bboxes for prompts
    QUERY_KEYS_PATH = "img_queries_keys.npy"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Retrieval Results
    print("Loading retrieval results...")
    try:
        retrieval_indices = np.load(RETRIEVAL_INDICES_PATH) # (num_queries, top_k)
        query_bboxes = np.load(METADATA_PATH) # (num_queries, 4)
        query_keys = np.load(QUERY_KEYS_PATH)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    num_queries, top_k = retrieval_indices.shape
    print(f"Processing {num_queries} queries with {top_k} candidates each.")

    # Load Model
    print("Loading SANSA model...")
    # Using arguments similar to the user's reference script
    # args.sam2_version, args.adaptformer_stages, args.channel_factor
    # We'll hardcode defaults or use a simple config object
    
    sam2_version = "tiny" # User mentioned SAM2-Tiny
    adaptformer_stages = [] # Default
    channel_factor = 4 # Default
    
    model = build_sansa(sam2_version, adaptformer_stages, channel_factor, DEVICE, obj_pred_scores=True)
    model.to(DEVICE)
    model.eval()
    
    # Transforms
    img_size = 518 # DINOv2 size, but SAM2 might use 1024. 
    # The reference script uses 518 for DINO but 512 or 1024 for SAM?
    # Reference script sansa_permir_obj_scores.py uses 518.
    
    def custom_to_tensor(pic):
        # Manual conversion to avoid torch.from_numpy version conflicts
        img_np = np.array(pic)
        # HWC to CHW
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
    
    for q_idx in tqdm(range(num_queries)):
        # 1. Load Query Image & Prompt
        q_key = query_keys[q_idx]
        # Safe key format used in extraction
        safe_q_key = q_key.replace('/', '_')
        q_img_path = os.path.join(QUERY_IMAGES_DIR, f"{safe_q_key}.jpg")
        
        if not os.path.exists(q_img_path):
            # Try to find it in candidates if it was part of core_db? 
            # No, queries are separate.
            # If missing, skip or use placeholder
            print(f"Warning: Query image {q_img_path} not found. Skipping.")
            reranked_indices.append(retrieval_indices[q_idx]) # Keep original order
            reranked_scores.append(np.zeros(top_k))
            continue
            
        query_img_pil = Image.open(q_img_path).convert('RGB')
        query_img = transform(query_img_pil).unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, C, H, W]
        
        # Prepare Prompt (BBox)
        # bbox is [x, y, w, h] -> need mask or box prompt?
        # Reference script uses `support_mask`.
        # We need to convert bbox to mask.
        bbox = np.array(query_bboxes[q_idx]) # [x, y, w, h]
        x, y, w, h = bbox
        
        # Create mask from bbox
        # Resize bbox to current img_size
        orig_w, orig_h = query_img_pil.size
        scale_x = img_size / orig_w
        scale_y = img_size / orig_h
        
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        
        # The prompt must be a bounding box
        # support_bbox = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32).to(DEVICE) # [1, 4]

        # The build_prompt_dict function expects a mask if we want to extract boxes from it,
        # OR we need to modify how we call it.
        # Looking at promptable_utils.py:
        # if prompt == 'box': boxes = get_bounding_boxes(frame_gt)
        # So it expects frame_gt to be a mask from which it extracts boxes.
        # We should provide a mask derived from our bbox.
        
        support_mask = torch.zeros((img_size, img_size), dtype=torch.float32)
        support_mask[y1:y2, x1:x2] = 1.0
        support_masks = support_mask.unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, H, W]
        #bbox = np.array([[x1, y1, x2, y2]]) # [1, 4]
        
        # Save a sample query with bbox visualization
        #if q_idx == 5:  # Save query 5 as example
            #import matplotlib.pyplot as plt
            #import matplotlib.patches as patches
            
            # Resize image for visualization
            #query_img_resized = query_img_pil.resize((img_size, img_size))
            
            #fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Resized image with scaled bbox
            #axes[0].imshow(query_img_resized)
            #rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            #axes[0].add_patch(rect)
            #axes[0].set_title(f'Resized Query at {img_size}x{img_size}\nBBox: [{x1}, {y1}, {x2}, {y2}]')
            #axes[0].axis('off')
            
            # Resized mask
            #axes[1].imshow(support_mask.cpu().numpy(), cmap='gray')
            #axes[1].set_title(f'Generated Mask\nBBox: [{x1}, {y1}, {x2}, {y2}]')
            #axes[1].axis('off')
            
            #plt.tight_layout()
            #plt.savefig('query_bbox_visualization.png', dpi=150, bbox_inches='tight')
            #plt.close()
            #print(f"\n[Saved visualization: query_bbox_visualization.png for Query {q_idx}]")
        
        # 2. Process Candidates
        candidate_indices = retrieval_indices[q_idx]
        scores = []
        
        # Batch processing for candidates
        BATCH_SIZE = 16 # Adjust based on GPU memory
        
        for i in range(0, top_k, BATCH_SIZE):
            #print(i)
            batch_indices = candidate_indices[i:i+BATCH_SIZE]
            
            # Load candidate images
            batch_imgs = []
            valid_batch_indices = []
            
            for cand_idx in batch_indices:
                cand_path = os.path.join(CANDIDATES_DIR, f"{cand_idx}.jpg")
                if os.path.exists(cand_path):
                    try:
                        img = Image.open(cand_path).convert('RGB')
                        img_t = transform(img).unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, C, H, W]
                        batch_imgs.append(img_t)
                        valid_batch_indices.append(True)
                    except Exception:
                        valid_batch_indices.append(False)
                else:
                    # Image not found (e.g. distractor not downloaded)
                    valid_batch_indices.append(False)
            
            if not batch_imgs:
                scores.extend([0.0] * len(batch_indices))
                continue
                
            # Concatenate: [Query, Cand1, Cand2, ...]
            # Model expects [1, T, C, H, W] where T = 1 (Query) + Batch (Candidates)
            # Actually, reference script does:
            # imgs = query_img
            # for b in range(batch_size): imgs = torch.cat([imgs, gallery_img], dim=1)
            
            imgs_tensor = torch.cat([query_img] + batch_imgs, dim=1).to(DEVICE)
            
            # Build Prompt Dict
            # n_shots=1 (just the query)
            # Note: prompt_type must match what rescale_prompt expects ('box' not 'bbox')
            # We pass support_masks because build_prompt_dict with 'box' expects a mask to extract boxes from
            prompt_dict = build_prompt_dict(bbox, "box", n_shots=1, train_mode=False, device=DEVICE)
            
            with torch.no_grad():
                outputs = model(imgs_tensor, prompt_dict)
                object_score_logits = outputs["object_score_logits"] # [T, 1]
                
            # Extract scores
            # Index 0 is query. Indices 1..Batch are candidates.
            # We need to map back to the original batch_indices (handling missing images)
            
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
                    batch_scores.append(0.0) # Low score for missing images
            
            scores.extend(batch_scores)
            
        # 3. Re-rank
        scores = np.array(scores)
        # Sort descending
        sorted_arg_indices = np.argsort(scores)[::-1]
        
        reranked_cand_indices = candidate_indices[sorted_arg_indices]
        reranked_cand_scores = scores[sorted_arg_indices]
        
        reranked_indices.append(reranked_cand_indices)
        reranked_scores.append(reranked_cand_scores)

        # Intermediate logging
        if q_idx < 5 or q_idx % 100 == 0:
            print(f"\n[Query {q_idx}] Processed. Valid candidates: {sum(1 for s in scores if s > 0)}")
            print(f"  Top 5 Scores: {reranked_cand_scores[:5]}")
            print(f"  Top 5 Indices: {reranked_cand_indices[:5]}")
        
    # Save Results
    print("Saving re-ranking results...")
    np.save("reranked_indices.npy", np.array(reranked_indices))
    np.save("reranked_scores.npy", np.array(reranked_scores))
    
    # Calculate mAP and Recall
    print("Calculating metrics...")
    try:
        # Load labels
        # Assuming labels are available as in ilias_retrieval.py
        # We need to know which indices are relevant for each query
        # Since we don't have the full ground truth matrix here easily, 
        # we can reuse the logic from ilias_retrieval.py if we load the labels.
        
        # Load labels
        q_labels = np.load("img_queries_labels.npy")
        db_labels = np.load("core_db_labels.npy")
        # Distractors have no labels or dummy labels? 
        # In ilias_retrieval, we concatenated db and distractor labels.
        # Distractors are usually not relevant, so their label is -1 or something distinct.
        # Let's assume distractors have label -1.
        
        # We need the total number of database items to reconstruct the labels array
        # From ilias_retrieval output: Total database size: 1004715
        # Core DB: 4715
        # Distractors: 1000000
        
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
            # Get labels of retrieved items
            retrieved_indices = reranked_indices[i] # Top-100
            retrieved_lbls = database_labels[retrieved_indices]
            
            # Relevance: 1 if label matches, 0 otherwise
            # Note: Distractors (-1) will never match valid query labels
            relevant = (retrieved_lbls == q_lbl).astype(int)
            
            # Total relevant in the entire database (needed for Recall)
            # This is expensive to compute every time if we search the whole array.
            # Optimization: Precompute counts for core_db labels.
            # Since distractors are irrelevant, total_relevant is just count in core_db.
            total_relevant = np.sum(db_labels == q_lbl)
            
            if total_relevant == 0:
                continue
                
            for k in k_values:
                if k > len(relevant):
                    rel_k = relevant
                else:
                    rel_k = relevant[:k]
                
                # Precision at k
                p_at_k = np.sum(rel_k) / k
                
                # Average Precision
                # AP = (1/min(m, k)) * sum(P@i * rel@i)
                # Standard definition: sum(P@i * rel@i) / total_relevant
                # But for @k metrics, usually we divide by min(total_relevant, k) or just total_relevant?
                # Let's use standard AP@k definition
                
                score = 0.0
                num_hits = 0.0
                
                for j in range(len(rel_k)):
                    if rel_k[j] == 1:
                        num_hits += 1
                        score += num_hits / (j + 1)
                
                ap = score / min(total_relevant, k)
                aps[k].append(ap)
                
                # Recall @ k
                recall = np.sum(rel_k) / total_relevant
                recalls[k].append(recall)
        
        print("\n=== Re-ranking Results ===")
        for k in k_values:
            map_k = np.mean(aps[k])
            mean_recall_k = np.mean(recalls[k])
            print(f"mAP@{k}: {map_k:.4f} | R@{k}: {mean_recall_k:.4f}")
            
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    main()
