r""" FSS-1000 few-shot semantic segmentation dataset """
import glob
import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms



class DatasetPerMIS(Dataset):
    def __init__(self, datapath, transform, split, shot):
        self.split = split
        self.benchmark = 'permis'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'PerMIRS/')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))

        self.class_ids = np.sort([int(cls_id) for cls_id in os.listdir(self.base_path)])

        self.img_metadata = self.build_img_metadata()
        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(class_sample, query_name)
        print(f"Loaded class {class_sample}")

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img, #tensor
                 'query_mask': query_mask, #tensor
                 'query_name': query_name, #str

                 'support_imgs': support_imgs, #tensor
                 'support_masks': support_masks, #tensor
                 'support_names': support_names, #list of str

                 'class_id': torch.tensor(class_sample)} #tensor
        #if self.split == 'test':
        #    batch['class_name'] = self.categories[class_sample-760]
        batch['class_name'] = batch['class_id'].item()
        
        return batch

    def load_frame(self, instance_idx, query_path):
        dataPath = self.base_path
        support_path = query_path.replace(f'/{os.path.basename(query_path)}', '/0.jpg')
        query_idx = int(os.path.basename(query_path).split('.')[0])

        query_img = Image.open(query_path).convert('RGB')
        support_imgs = [Image.open(support_path).convert('RGB')]
        maskDir = os.path.join(dataPath, f"{instance_idx}/masks.npz.npy")
        masks_np = np.load(maskDir, allow_pickle=True)



        query_mask = self.read_mask(np.array(list(masks_np[query_idx].values()))[0])
        support_masks = [self.read_mask(np.array(list(masks_np[0].values()))[0])]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, mask):
        mask = torch.tensor(mask)
        mask[mask == True] = 1
        mask[mask == False] = 0
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx] 
        instance_sample = int(query_name.split('/')[-2])
        support_names = [query_name.replace(f'/{query_name.split("/")[-1]}', '/0.jpg')]

        return query_name, support_names, instance_sample

    
    def build_img_metadata(self):
        img_metadata = []
        for idx in self.class_ids:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, str(idx)))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg' and '0' not in os.path.basename(img_path).split('.')[0]:
                    img_metadata.append(img_path)
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
    dataset = DatasetPerMIS(datapath=args.data_root, transform=transform,
                 shot=args.shots, split=image_set)

    return dataset
