r""" FSS-1000 few-shot semantic segmentation dataset """
import glob
import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms



class DatasetPerSeg(Dataset):
    def __init__(self, datapath, transform, split, shot):
        self.split = split
        self.benchmark = 'perseg'
        self.shot = shot

        self.base_path = os.path.join('/home/arvc/Marcos/INVESTIGACION/Personalize-SAM-main/PerSeg/')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))

        self.class_ids = sorted(os.listdir(f"{self.base_path}/Images/"))
        
        self.img_metadata = self.build_img_metadata()
        print(f"Number of images in {split} split: {len(self.img_metadata)}")
        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name)

        query_img = self.transform(query_img)
        # query_mask viene con shape [1, H, W] de read_mask
        # Si el tamaño no coincide con query_img, interpolar
        if query_mask.shape[-2:] != query_img.shape[-2:]:
            query_mask = F.interpolate(query_mask.unsqueeze(0).float(), size=query_img.shape[-2:], mode='nearest').squeeze(0)
        # Remover dimensión de canal si existe: [1, H, W] -> [H, W]
        query_mask = query_mask.squeeze(0) if query_mask.dim() == 3 else query_mask

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            # smask viene con shape [1, H, W] de read_mask
            if smask.shape[-2:] != support_imgs.shape[-2:]:
                smask = F.interpolate(smask.unsqueeze(0).float(), size=support_imgs.shape[-2:], mode='nearest').squeeze(0)
            smask = smask.squeeze(0) if smask.dim() == 3 else smask
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img, #tensor
                 'query_mask': query_mask, #tensor
                 'query_name': query_name, #str

                 'support_imgs': support_imgs, #tensor
                 'support_masks': support_masks, #tensor
                 'support_names': support_names, #list of str
                 'class_name': query_name.split('/')[-2]
        }
                 #'class_id': torch.tensor(class_sample)} #tensor
        #if self.split == 'test':
        #    batch['class_name'] = self.categories[class_sample-760]
        #batch['class_name'] = batch['class_id'].item()
        
        return batch

    def load_frame(self, query_path):
        support_path = query_path.replace(f'/{os.path.basename(query_path)}', '/00.jpg')
        query_img = Image.open(query_path).convert('RGB')
        support_imgs = [Image.open(support_path).convert('RGB')]

        query_mask_path = query_path.replace('Images', 'Annotations').replace('.jpg', '.png')
        support_mask_path = support_path.replace('Images', 'Annotations').replace('.jpg', '.png')

        query_mask = self.read_mask(query_mask_path)
        support_masks = [self.read_mask(support_mask_path)]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, mask_path):
        mask = Image.open(mask_path).convert('L')  # Convertir a escala de grises
        # No usar self.transform porque incluye normalización RGB
        # Solo redimensionar y convertir a tensor
        mask = transforms.functional.resize(mask, (518, 518), interpolation=transforms.InterpolationMode.NEAREST)
        mask = transforms.functional.to_tensor(mask)
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx] 
        #instance_sample = int(query_name.split('/')[-2])
        support_names = [query_name.replace(f'/{query_name.split("/")[-1]}', '/00.jpg')]

        return query_name, support_names

    
    def build_img_metadata(self):
        img_metadata = []
        for class_id in self.class_ids:
            if class_id.startswith('.'):
                continue
            img_paths = os.listdir(os.path.join(self.base_path, 'Images', class_id))
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg' and '00' not in os.path.basename(img_path).split('.')[0]:
                    img_metadata.append(f"{self.base_path}/Images/{class_id}/{img_path}")
        return img_metadata
    

def build(image_set, args):
    img_size = 518
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image_set == 'train':
        image_set = 'trn'
    elif image_set == 'val':
        image_set = 'test'
    dataset = DatasetPerSeg(datapath=args.data_root, transform=transform,
                 shot=args.shots, split=image_set)

    return dataset
