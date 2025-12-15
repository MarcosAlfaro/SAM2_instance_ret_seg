import numpy as np
import faiss
import time

def main():
    # 1. Load features
    print("Loading features from disk")
    try:
        # Assuming files are in the current directory as seen in the file list
        q_feats = np.load("img_queries_features_dinov2_small.npy")
        db_feats = np.load("core_db_features_dinov2_small.npy")
        dist_feats = np.load("distractors_1m_features_dinov2_small.npy")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print(f"Queries shape: {q_feats.shape}")
    print(f"Core DB shape: {db_feats.shape}")
    print(f"Distractors shape: {dist_feats.shape}")

    # Ensure float32 for Faiss
    if q_feats.dtype != np.float32:
        q_feats = q_feats.astype(np.float32)
    if db_feats.dtype != np.float32:
        db_feats = db_feats.astype(np.float32)
    if dist_feats.dtype != np.float32:
        dist_feats = dist_feats.astype(np.float32)

    # 2. Prepare Database
    database_features = np.vstack([db_feats, dist_feats])
    d = database_features.shape[1]
    print(f"Total database size: {database_features.shape[0]} vectors of dimension {d}")

    # 3. Build Faiss Index on GPU
    print("Building Faiss index on GPU")
    res = faiss.StandardGpuResources()
    
    # IndexFlatIP implements inner product (cosine similarity if vectors are normalized)
    index_flat = faiss.IndexFlatIP(d)
    
    # Move index to GPU
    # If you have multiple GPUs, you can use index_cpu_to_all_gpus
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)

    start_time = time.time()
    gpu_index.add(database_features)
    print(f"Index built in {time.time() - start_time:.2f} seconds")
    print(f"Index contains {gpu_index.ntotal} vectors")

    # 4. Search
    k = 100 # Top-k candidates
    print(f"Searching for {q_feats.shape[0]} queries (top {k})")
    
    start_time = time.time()
    D, I = gpu_index.search(q_feats, k)
    print(f"Search done in {time.time() - start_time:.2f} seconds")

    # 5. Compute mAP
    # Save indices for re-ranking
    np.save("retrieval_indices_top100.npy", I)
    print("Saved retrieval_indices_top100.npy")

    print("Loading labels for mAP calculation")
    try:
        q_labels = np.load("img_queries_labels.npy")
        db_labels = np.load("core_db_labels.npy")
        
        print("Computing mAP and R@k")
        core_db_len = db_feats.shape[0]
        aps = []
        recalls = []
        
        # Pre-compute total relevant items for each unique label
        unique_labels = np.unique(q_labels)
        label_counts = {label: np.sum(db_labels == label) for label in unique_labels}
        
        for i in range(len(q_labels)):
            q_label = q_labels[i]
            total_relevant = label_counts.get(q_label, 0)
            
            if total_relevant == 0:
                aps.append(0.0)
                recalls.append(0.0)
                continue
                
            retrieved_indices = I[i]
            
            # Identify relevant items
            # Indices < core_db_len are core items, check label
            # Indices >= core_db_len are distractors (irrelevant)
            
            is_core = retrieved_indices < core_db_len
            is_relevant = np.zeros(len(retrieved_indices), dtype=bool)
            
            # Check labels for core items
            core_indices = retrieved_indices[is_core]
            if len(core_indices) > 0:
                matches = db_labels[core_indices] == q_label
                is_relevant[is_core] = matches
            
            # Compute Recall
            recall = np.sum(is_relevant) / total_relevant
            recalls.append(recall)

            # Compute Precision at each rank
            cumulative_relevant = np.cumsum(is_relevant)
            ranks = np.arange(1, len(retrieved_indices) + 1)
            precisions = cumulative_relevant / ranks
            
            # AP = sum(precision * rel) / total_relevant
            ap = np.sum(precisions * is_relevant) / total_relevant
            aps.append(ap)
            
        mAP = np.mean(aps)
        mRecall = np.mean(recalls)
        print(f"mAP@{k}: {mAP:.4f}")
        print(f"R@{k}: {mRecall:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error loading labels for mAP: {e}")

    # 6. Save results for re-ranking
    print("Saving retrieval results for re-ranking")
    np.save("retrieval_indices.npy", I)
    np.save("retrieval_scores.npy", D)
    print("Saved retrieval_indices.npy and retrieval_scores.npy")

if __name__ == "__main__":
    main()
