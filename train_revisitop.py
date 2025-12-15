import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from PIL import Image
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from collections import defaultdict

from models.sansa.sansa import build_sansa

class TripletLoss(nn.Module):
    """Triplet loss for object score optimization"""
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor_scores, positive_scores, negative_scores):
        """
        Args:
            anchor_scores: Not used (dummy for compatibility)
            positive_scores: Object scores for positive (same landmark) images [B]
            negative_scores: Object scores for negative (different landmark) images [B]
        """
        # We want positive_scores > negative_scores
        # Loss = max(0, margin + negative_score - positive_score)
        loss = torch.clamp(self.margin + negative_scores - positive_scores, min=0.0)
        return loss.max()


class RevisitOxfordParisDataset(Dataset):
    """Dataset for Revisit Oxford & Paris with real bounding boxes"""
    
    def __init__(self, dataset_name='roxford5k', img_size=518, use_both=True, triplets_per_epoch=1000):
        """
        Args:
            dataset_name: 'roxford5k' or 'rparis6k' or 'both'
            img_size: target image size
            use_both: if True, combines both datasets
            triplets_per_epoch: number of triplet samples to generate per epoch
        """
        self.img_size = img_size
        self.triplets_per_epoch = triplets_per_epoch
        self.datasets = []
        
        base_path = "/home/arvc/Marcos/INVESTIGACION/revisitop-master/data/datasets"
        
        datasets_to_load = []
        if dataset_name == 'both' or use_both:
            datasets_to_load = ['roxford5k', 'rparis6k']
        else:
            datasets_to_load = [dataset_name]
        
        print(f"Loading Revisit Oxford & Paris datasets: {datasets_to_load}")
        
        self.samples = []
        self.label_to_indices = defaultdict(list)
        
        for dname in datasets_to_load:
            pkl_path = os.path.join(base_path, dname, f"gnd_{dname}.pkl")
            img_dir = os.path.join(base_path, dname, "jpg")
            
            print(f"\nLoading {dname}...")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract data
            gnd = data['gnd']  # List of dicts with 'bbx', 'easy', 'hard', 'junk'
            imlist = data['imlist']  # Database image names (without .jpg)
            qimlist = data['qimlist']  # Query image names (without .jpg)
            
            print(f"  Queries: {len(qimlist)}")
            print(f"  Database images: {len(imlist)}")
            
            # Process each query
            for q_idx, q_name in enumerate(qimlist):
                q_path = os.path.join(img_dir, f"{q_name}.jpg")
                if not os.path.exists(q_path):
                    continue
                
                # Get query bounding box
                q_bbox = gnd[q_idx]['bbx']  # [x1, y1, x2, y2] format
                
                # Get positive indices (easy + hard)
                positive_indices = gnd[q_idx]['easy'] + gnd[q_idx]['hard']
                
                if len(positive_indices) == 0:
                    continue
                
                # IMPORTANT: Only add QUERY as anchor sample with bbox
                # Database images have different dimensions, so query bbox doesn't fit them
                label = f"{dname}_{q_idx}"  # Unique label per query
                sample_idx = len(self.samples)
                
                self.samples.append({
                    'image_path': q_path,
                    'bbox': q_bbox,
                    'label': label,
                    'is_query': True,
                    'dataset': dname 
                })
                self.label_to_indices[label].append(sample_idx)
                
                # Add positive images WITHOUT bbox (they will be used as gallery images only)
                for pos_idx in positive_indices:
                    if pos_idx >= len(imlist):
                        continue
                    
                    pos_name = imlist[pos_idx]
                    pos_path = os.path.join(img_dir, f"{pos_name}.jpg")
                    
                    if not os.path.exists(pos_path):
                        continue
                    
                    sample_idx = len(self.samples)
                    self.samples.append({
                        'image_path': pos_path,
                        'bbox': None,  # No bbox for database images
                        'label': label,
                        'is_query': False,
                        'dataset': dname
                    })
                    self.label_to_indices[label].append(sample_idx)
        
        # Filter labels with at least 2 samples (needed for triplets)
        self.valid_labels = [lbl for lbl, indices in self.label_to_indices.items() 
                            if len(indices) >= 2]
        
        # Store query indices (anchors with valid bboxes)
        self.query_indices = [i for i, s in enumerate(self.samples) if s['is_query']]
        
        print(f"\n=== Dataset Summary ===")
        print(f"Total samples: {len(self.samples)}")
        print(f"Query images (anchors): {len(self.query_indices)}")
        print(f"Valid labels (queries with positives): {len(self.valid_labels)}")
        print(f"Average positives per query: {np.mean([len(self.label_to_indices[l]) for l in self.valid_labels]):.1f}")
        print(f"Triplets per epoch: {self.triplets_per_epoch}")
        
        # Transforms
        def custom_to_tensor(pic):
            img_np = np.array(pic)
            img = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
            return img / 255.0
        
        self.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.Lambda(custom_to_tensor),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.triplets_per_epoch
    
    def load_sample(self, idx):
        """Load and process a single sample"""
        sample = self.samples[idx]
        
        try:
            # Load image
            img = Image.open(sample['image_path']).convert('RGB')
            orig_w, orig_h = img.size
            
            # Scale bbox coordinates (only for query images)
            if sample['bbox'] is not None:
                x1, y1, x2, y2 = sample['bbox']
                
                # Scale bbox to target size
                scale_x = self.img_size / orig_w
                scale_y = self.img_size / orig_h
                
                scaled_bbox = [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ]
                
                # Clamp bbox
                scaled_bbox[0] = max(0, min(scaled_bbox[0], self.img_size - 1))
                scaled_bbox[1] = max(0, min(scaled_bbox[1], self.img_size - 1))
                scaled_bbox[2] = max(scaled_bbox[0] + 1, min(scaled_bbox[2], self.img_size))
                scaled_bbox[3] = max(scaled_bbox[1] + 1, min(scaled_bbox[3], self.img_size))
            else:
                # Database images don't have bbox
                scaled_bbox = None
            
            # Transform image (resize to 518x518)
            img_tensor = self.transform(img)
            
            return {
                'image': img_tensor,
                'bbox': scaled_bbox,
                'label': sample['label']
            }
        except Exception as e:
            return None
    
    def __getitem__(self, idx):
        """Sample a triplet: anchor, positive, negative"""
        
        # Randomly select an anchor from query images (idx is ignored)
        anchor_idx = random.choice(self.query_indices)
        anchor_sample = self.samples[anchor_idx]
        anchor_label = anchor_sample['label']
        
        # Get indices for this label
        same_label_indices = self.label_to_indices[anchor_label]
        
        # Sample positive (different from anchor)
        if len(same_label_indices) < 2:
            return None
        
        positive_indices = [i for i in same_label_indices if i != idx]
        if len(positive_indices) == 0:
            return None
        
        pos_idx = random.choice(positive_indices)
        
        # Sample negative (different label)
        if len(self.valid_labels) < 2:
            return None
        
        neg_label = random.choice([l for l in self.valid_labels if l != anchor_label])
        neg_idx = random.choice(self.label_to_indices[neg_label])
        
        # Load all three
        anchor_data = self.load_sample(anchor_idx)
        positive_data = self.load_sample(pos_idx)
        negative_data = self.load_sample(neg_idx)
        
        if anchor_data is None or positive_data is None or negative_data is None:
            return None
        
        return {
            'anchor': anchor_data,
            'positive': positive_data,
            'negative': negative_data
        }


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data in tqdm(dataloader, desc="Training"):
        if batch_data is None:
            continue
        
        batch_losses = []
        
        for triplet in batch_data:
            if triplet is None:
                continue
            
            try:
                # Get anchor image and prompt (support set)
                anc_img = triplet['anchor']['image'].unsqueeze(0).to(device)  # [1, C, H, W]
                anc_bbox = triplet['anchor']['bbox']
                
                # Get positive and negative images (gallery set)
                pos_img = triplet['positive']['image'].unsqueeze(0).to(device)  # [1, C, H, W]
                neg_img = triplet['negative']['image'].unsqueeze(0).to(device)  # [1, C, H, W]
                
                # Stack all three images: [1, 3, C, H, W] -> [anchor, positive, negative]
                # This creates a batch where:
                # - Frame 0 (anchor): support with prompt -> stored in memory
                # - Frame 1 (positive): gallery without prompt -> queries memory
                # - Frame 2 (negative): gallery without prompt -> queries memory
                imgs = torch.stack([anc_img, pos_img, neg_img], dim=1)  # [1, 3, C, H, W]
                
                # Create SAM2 prompt for anchor only (n_shots=1)
                x1, y1, x2, y2 = anc_bbox
                point_coords = torch.tensor([[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device)
                point_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
                
                prompt_dict = {
                    'shots': 1,  # Only the first frame (anchor) is support
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
                
                outputs = model(imgs, prompt_dict)
                
                # Extract object scores
                object_scores = torch.sigmoid(outputs['object_score_logits'])  # [3, 1]
                anc_score = object_scores[0]
                pos_score = object_scores[1]
                neg_score = object_scores[2]
                
                # Compute loss
                loss = criterion(
                    anchor_scores=anc_score, 
                    positive_scores=pos_score,
                    negative_scores=neg_score
                )
                
                batch_losses.append(loss)
                
            except Exception as e:
                print(f"Error in forward pass: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(batch_losses) == 0:
            continue
        
        batch_loss = torch.stack(batch_losses).mean()
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        

        if num_batches % 100 == 0:
            if len(batch_losses) > 0:
                print(f"\nBatch {num_batches} - Scores: Anc={anc_score.item():.4f}, Pos={pos_score.item():.4f}, Neg={neg_score.item():.4f}, Loss={batch_loss.item():.4f}")
        num_batches += 1   
    return total_loss / max(num_batches, 1)


def main():
    # Configuration
    IMG_SIZE = 518
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    CHECKPOINT_DIR = "checkpoints"
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # Load model
    print("\nLoading SANSA model...")
    model = build_sansa("tiny", [], 4, DEVICE, obj_pred_scores=True)
    model.to(DEVICE)
    
    print("Freezing SAM2 backbone...")
    for name, param in model.named_parameters():
        # Unfreeze only the object score prediction head
        if 'iou' in name or 'score' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Load dataset
    print("\nLoading Revisit Oxford & Paris dataset...")
    dataset = RevisitOxfordParisDataset(dataset_name='both', img_size=IMG_SIZE, use_both=True)
    
    # Dataloader
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return batch
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # Training setup
    criterion = TripletLoss(margin=0.2)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, DEVICE)
        
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"sansa_revisitop_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
