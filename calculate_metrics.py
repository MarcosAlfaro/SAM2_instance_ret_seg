import numpy as np
import os

def main():
    print("Loading re-ranking results")
    reranked_indices = np.load("npy_files/reranked_indices.npy")        
    q_labels = np.load("npy_files/img_queries_labels.npy")
    db_labels = np.load("npy_files/core_db_labels.npy")

    num_queries = len(reranked_indices)
    print(f"Loaded results. Num. queries: {num_queries}")

    # Prepare Database Labels
    # DB: 4715 Distractors: 1000000
    num_distractors = 1000000
    dist_labels = np.full(num_distractors, -1, dtype=db_labels.dtype)
    database_labels = np.concatenate([db_labels, dist_labels])
    
    # Calculate mAP@k
    k_values = [1, 5, 10, 20, 50, 100]
    
    # 1. Calculate for Re-ranking (SAM2)
    print("\nCalculating metrics for Re-ranking (SAM2)")
    calculate_and_print_metrics(reranked_indices, q_labels, database_labels, db_labels, k_values, "Re-ranking (SAM2)")

    # 2. Calculate for Initial Retrieval (DINOv2)
    print("\nCalculating metrics (DINOv2)")
    dino_indices = np.load("retrieval_indices_top100.npy")
    calculate_and_print_metrics(dino_indices, q_labels, database_labels, db_labels, k_values, "Initial Retrieval (DINOv2)")

def calculate_and_print_metrics(indices, q_labels, database_labels, db_labels, k_values, method_name):
    aps = {k: [] for k in k_values}
    recalls = {k: [] for k in k_values}
    
    num_queries = len(indices)
    
    for i in range(num_queries):
        q_lbl = q_labels[i]
        retrieved_indices = indices[i] # Top-100        
        retrieved_lbls = database_labels[retrieved_indices]
        
        positives = (retrieved_lbls == q_lbl).astype(int)
        
        # Total positives in the entire database
        total_positives = np.sum(db_labels == q_lbl)
        
        if total_positives == 0:
            continue
            
        for k in k_values:
            if k > len(positives):
                rel_k = positives
            else:
                rel_k = positives[:k]
            
            # Average Precision
            score = 0.0
            num_hits = 0.0
            
            for j in range(len(rel_k)):
                if rel_k[j] == 1:
                    num_hits += 1
                    score += num_hits / (j + 1)
            
            ap = score / min(total_positives, k)
            aps[k].append(ap)
            
            # Recall @ k
            recall = np.sum(rel_k) / total_positives
            recalls[k].append(recall)
    
    print(f"\n=== {method_name} Results ===")
    for k in k_values:
        map_k = np.mean(aps[k])
        mean_recall_k = np.mean(recalls[k])
        print(f"mAP@{k}: {map_k:.4f} | R@{k}: {mean_recall_k:.4f}")

if __name__ == "__main__":
    main()
