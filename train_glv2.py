import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from collections import defaultdict

# Intentar importar tu modelo (Asegúrate que la ruta sea correcta)
try:
    from models.sansa.sansa_original import build_sansa
except ImportError:
    # Dummy class para pruebas si no tienes el archivo a mano al copiar/pegar
    print("Warning: Could not import build_sansa. Using dummy model for syntax check.")
    class build_sansa(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x, p): return {'object_score_logits': torch.randn(2, 1)}

# --- CONFIGURACIÓN DEL CSV ---
# Este CSV debe tener: 'id', 'url', 'landmark_id' y OPCIONALMENTE 'bounding_boxes'
# Si no tiene bounding_boxes, se generarán cajas aleatorias
CSV_PATH = "train.csv" 
# Si estás probando y no tienes el CSV, pon esto en True para generar uno falso
GENERATE_DUMMY_CSV = False 
USE_RANDOM_BOXES = True  # Set to True if CSV doesn't have bounding boxes 

class TripletLoss(nn.Module):
    """Triplet loss for object score optimization"""
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor_scores, positive_scores, negative_scores):
        loss = torch.clamp(self.margin + negative_scores - positive_scores, min=0.0)
        return loss.mean() # Changed to mean for stability

def download_image(url):
    """Download image from URL directly into RAM"""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception:
        return None

class GLDv2CSVDataset(Dataset):
    """Dataset using a local CSV with 'url', 'landmark_id' and optionally 'bounding_boxes'"""
    
    def __init__(self, csv_path, img_size=518, max_samples=None, use_random_boxes=True):
        self.img_size = img_size
        self.use_random_boxes = use_random_boxes
        
        print(f"Loading Metadata from {csv_path}...")
        try:
            # Alternative CSV reading to avoid pandas numpy conversion bug
            # Use pure Python data structures
            import csv
            data_list = []
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if max_samples and i >= max_samples:
                        break
                    # Only keep rows with valid URL and landmark_id
                    if row.get('url') and row.get('landmark_id'):
                        data_list.append(row)
                    if (i + 1) % 10000 == 0:
                        print(f"Loaded {i + 1} rows...")
            
            # Store as list instead of DataFrame to avoid pandas bug
            self.data = data_list
            
            print(f"Loaded {len(self.data)} valid rows")
            if len(self.data) > 0:
                print(f"Columns: {list(self.data[0].keys())}")
            
            #Check if bounding_boxes column exists
            has_boxes = len(self.data) > 0 and 'bounding_boxes' in self.data[0]
            
            if has_boxes:
                print("Found bounding_boxes column - will use real annotations")
                self.use_random_boxes = False
            else:
                print("No bounding_boxes column - will generate random boxes")
                self.use_random_boxes = True
            
            print(f"Building landmark index...")
            # Group by landmark_id using pure Python
            self.id_to_indices = {}
            for idx, row in enumerate(self.data):
                landmark_id = row['landmark_id']
                if landmark_id not in self.id_to_indices:
                    self.id_to_indices[landmark_id] = []
                self.id_to_indices[landmark_id].append(idx)
            
            # Filter landmarks with at least 2 samples
            self.valid_ids = [lid for lid, indices in self.id_to_indices.items() if len(indices) >= 2]
            print(f"Valid landmark classes with >=2 samples: {len(self.valid_ids)}")
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise e

        # Transformación base (sin crop, el resize lo hacemos después de leer la caja)
        # Nota: Hacemos el resize manual en __getitem__ para ajustar la caja proporcionalmente
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def parse_box(self, box_str):
        """
        Parsea el string de la caja.
        Formato esperado en GLD Kaggle: "x1 y1 x2 y2" (str) o "confidence x1 y1 x2 y2"
        """
        try:
            parts = str(box_str).split()
            # A veces vienen con confidence al principio, cogemos los últimos 4
            coords = [float(x) for x in parts[-4:]] 
            return coords # [x1, y1, x2, y2]
        except:
            return None
    
    def generate_random_box(self, img_size):
        """Generate random bounding box for training"""
        bbox_size_ratio = random.uniform(0.4, 0.8)
        bbox_w = int(img_size * bbox_size_ratio)
        bbox_h = int(img_size * bbox_size_ratio)
        
        # Random aspect ratio variation (±20%)
        aspect_ratio_factor = random.uniform(0.8, 1.2)
        bbox_h = int(bbox_h * aspect_ratio_factor)
        
        # Ensure bbox fits in image
        bbox_w = min(bbox_w, img_size)
        bbox_h = min(bbox_h, img_size)
        
        # Random position
        max_x = img_size - bbox_w
        max_y = img_size - bbox_h
        x1 = random.randint(0, max(1, max_x))
        y1 = random.randint(0, max(1, max_y))
        x2 = x1 + bbox_w
        y2 = y1 + bbox_h
        
        return [x1, y1, x2, y2]

    def resize_image_and_box(self, img, box=None):
        """Redimensiona la imagen y ajusta las coordenadas de la caja"""
        w, h = img.size
        target_size = self.img_size
        
        # Resize imagen
        img_resized = img.resize((target_size, target_size), Image.BICUBIC)
        
        if box is None:
            # Generate random box after resize
            new_box = self.generate_random_box(target_size)
        else:
            # Ajustar caja existente
            x1, y1, x2, y2 = box
            scale_x = target_size / w
            scale_y = target_size / h
            
            new_box = [
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            ]
            
            # Clamp para seguridad
            new_box[0] = max(0, new_box[0])
            new_box[1] = max(0, new_box[1])
            new_box[2] = min(target_size, new_box[2])
            new_box[3] = min(target_size, new_box[3])
        
        return img_resized, new_box

    def __len__(self):
        return len(self.data)
    
    def get_sample_at_idx(self, idx):
        """Helper to get data from a specific row index"""
        row = self.data[idx]  # Now accessing list, not DataFrame
        url = row['url']
        
        img = download_image(url)
        if img is None: return None
        
        # Get bounding box
        if self.use_random_boxes:
            # No box in CSV, will be generated after resize
            img, box = self.resize_image_and_box(img, box=None)
        else:
            # Parse box from CSV
            box_str = row['bounding_boxes']
            box = self.parse_box(box_str)
            if box is None: return None
            # Resize Image and Box coordinates
            img, box = self.resize_image_and_box(img, box)
        
        return {'image': img, 'box': box, 'id': row['landmark_id']}

    def __getitem__(self, idx):
        # 1. ANCHOR (La fila actual)
        anchor_data = self.get_sample_at_idx(idx)
        if anchor_data is None: return None
        
        anchor_label = anchor_data['id']
        
        # 2. POSITIVE (Mismo ID, diferente imagen)
        possible_pos_indices = self.id_to_indices[anchor_label]
        # Filtrar el índice del anchor si hay suficientes muestras
        if len(possible_pos_indices) > 1:
            possible_pos_indices = [x for x in possible_pos_indices if x != idx]
            
        if len(possible_pos_indices) == 0: return None
        
        pos_idx = np.random.choice(possible_pos_indices)
        positive_data = self.get_sample_at_idx(pos_idx)
        if positive_data is None: return None
        
        # 3. NEGATIVE (Diferente ID)
        # Intentamos hasta encontrar uno válido
        for _ in range(3):
            neg_label = np.random.choice(self.valid_ids)
            while neg_label == anchor_label:
                neg_label = np.random.choice(self.valid_ids)
            
            neg_idx = np.random.choice(self.id_to_indices[neg_label])
            negative_data = self.get_sample_at_idx(neg_idx)
            if negative_data is not None:
                break
        else:
            return None # Falló en encontrar negativo válido
            
        return {
            'anchor': anchor_data,
            'positive': positive_data,
            'negative': negative_data
        }

def process_sample(sample, normalize_transform, device):
    """
    Procesa la muestra usando la CAJA REAL del CSV.
    """
    img = sample['image'] # PIL ya redimensionada
    bbox = sample['box']  # [x1, y1, x2, y2] ya reescalado
    
    # Convertir a Tensor y Normalizar
    img_np = np.array(img)
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = normalize_transform(img_tensor).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, C, H, W]
    
    # Preparar Prompt para SAM2 usando la caja real
    x1, y1, x2, y2 = bbox
    
    # Crear Prompt Box
    point_coords = torch.tensor([[[x1, y1], [x2, y2]]], dtype=torch.float32, device=device) # [1, 2, 2]
    point_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device) # 2=TopLeft, 3=BottomRight
    
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
    
    return img_tensor, prompt_dict

def train_epoch(model, dataloader, criterion, optimizer, device, normalize_transform):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data in tqdm(dataloader, desc="Training"):
        if batch_data is None: continue
        
        batch_losses = []
        
        for triplet in batch_data:
            if triplet is None: continue
            
            try:
                # Ahora pasamos 'triplet['anchor']' que contiene {'image', 'box'}
                # process_sample extraerá esa caja específica
                anc_img, anc_prompt = process_sample(triplet['anchor'], normalize_transform, device)
                pos_img, pos_prompt = process_sample(triplet['positive'], normalize_transform, device)
                neg_img, neg_prompt = process_sample(triplet['negative'], normalize_transform, device)
                
                # --- FORWARD PASS (Optimizado) ---
                # Anchor Score
                anc_out = model(anc_img, anc_prompt)
                anc_score = torch.sigmoid(anc_out['object_score_logits'][0])
                
                # Positive Score
                # Truco: SAM2 a veces permite batching en prompt, pero aquí vamos 1 a 1 por seguridad
                # Si tu modelo soporta batches puros, puedes apilar las imágenes antes.
                pos_out = model(pos_img, pos_prompt) # Usamos el prompt POSITIVO para la imagen positiva
                # NOTA: En tu código original pasabas 'anchor_prompt' a la imagen positiva. 
                # Eso está MAL para GLDv2 porque la caja del anchor no coincide con la foto del positivo.
                # Aquí corregido: cada imagen lleva SU propia caja.
                pos_score = torch.sigmoid(pos_out['object_score_logits'][0])
                
                # Negative Score
                neg_out = model(neg_img, neg_prompt)
                neg_score = torch.sigmoid(neg_out['object_score_logits'][0])
                
                loss = criterion(anc_score, pos_score, neg_score)
                batch_losses.append(loss)
                
            except Exception as e:
                # print(f"Error in forward: {e}") 
                continue
        
        if len(batch_losses) == 0: continue
        
        batch_loss = torch.stack(batch_losses).mean()
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def main():
    # --- Generación de CSV Falso para Testing (Solo si no tienes el archivo) ---
    if GENERATE_DUMMY_CSV:
        print("Generating dummy train.csv for testing...")
        data = {
            'id': [f'img_{i}' for i in range(20)],
            'url': ['https://via.placeholder.com/500']*20,
            'landmark_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7],
            'bounding_boxes': ['10 10 100 100'] * 20
        }
        pd.DataFrame(data).to_csv(CSV_PATH, index=False)

    IMG_SIZE = 518
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4 # Reducido porque descargar URLs consume threads
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    
    # 1. Modelo
    print("Building Model...")
    model = build_sansa("tiny", [], 4, DEVICE, obj_pred_scores=True)
    model.to(DEVICE)
    
    for name, param in model.named_parameters():
        if 'iou' in name or 'score' in name: # Asegurar nombres correctos del head
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # 2. Dataset
    # Aquí es donde ocurre la magia del CSV
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encuentra {CSV_PATH}. Por favor descarga el train.csv de Google Landmarks.")

    dataset = GLDv2CSVDataset(csv_path=CSV_PATH, img_size=IMG_SIZE, max_samples=50000, use_random_boxes=USE_RANDOM_BOXES)
    
    # 3. Dataloader
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0: return None
        return batch

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, collate_fn=collate_fn)
    
    criterion = TripletLoss(margin=0.2)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # 4. Loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}")
        loss = train_epoch(model, dataloader, criterion, optimizer, DEVICE, normalize_transform)
        print(f"Loss: {loss}")
        
        # Save
        torch.save(model.state_dict(), f"checkpoints/sansa_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()