import PIL
import numpy as np
import torch
import os
from sklearn.metrics import average_precision_score
from torchvision.transforms import PILToTensor
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import gc
from PerMIRS.visualization_utils import bbox_from_mask
import argparse
from util.commons import setup_logging, make_deterministic, resume_from_checkpoint
from models.sansa.sansa_original import build_sansa
import opts
from util.promptable_utils import build_prompt_dict
from PIL import Image
import time


def convert_mask_values(mask):
        mask = mask.cpu().detach().numpy()
        mask[mask >= 0.0] = 255
        mask[mask < 0.0] = 0
        mask = Image.fromarray(mask.astype(np.uint8))
        return mask


def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    print(args)

    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device, obj_pred_scores=True)
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)

    print(f"number of params: {n_parameters}")
    print('Start inference')

    eval_permir(model, args)
    return 


def center_crop(im, min_obj_x=None, max_obj_x=None, offsets=None):
    if offsets is None:
        width, height = im.size  # Get dimensions
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2

        if min_obj_x < left:
            diff = abs(left - min_obj_x)
            left = min_obj_x
            right = right - diff
        if max_obj_x > right:
            diff = abs(right - max_obj_x)
            right = max_obj_x
            left = left + diff
    else:
        left, top, right, bottom = offsets

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im, (left, top, right, bottom)


def load_im_into_format_from_path(im_path, size=512, offsets=None):
    im, offsets = center_crop(PIL.Image.open(im_path), offsets=offsets)
    return im


def load_im_into_format_from_image(image, size=512, min_obj_x=None, max_obj_x=None):
    im, offsets = center_crop(image, min_obj_x=min_obj_x, max_obj_x=max_obj_x)
    return im.resize((size, size)), offsets

def read_mask(mask):
    mask = torch.tensor(mask)
    mask[mask == True] = 1
    mask[mask == False] = 0
    return mask


def eval_permir(model, args):

    img_size = 512
    attn_size = 64
    dift_size = 512

    dataset_dir = "/home/arvc/Marcos/INVESTIGACION/PDM-main/PerMIRS/"

    if "ret_data_sansa.pt" not in os.listdir(dataset_dir):
        # organize data for retrieval
        query_frames = []
        query_labels = []
        query_dift_features = []
        query_dift_mask = []
        gallery_frames = []
        gallery_labels = []
        gallery_masks = []

        for vid_idx, vid_id in tqdm(enumerate(sorted(os.listdir(dataset_dir)))):
            
            try:
                torch.cuda.empty_cache()
                gc.collect()
                
                
                masks_attn = []
                masks_dift = []
                masks_relative_size = []
                
                # Cargar datos de forma más eficiente
                masks_np = np.load(f"{dataset_dir}/{vid_id}/masks.npz.npy", allow_pickle=True)
                
                frame_paths = []
                
                for f in range(3):
                    # Procesar cada frame individualmente para evitar acumulación de memoria
                    xmin, ymin, xmax, ymax = [int(x) for x in bbox_from_mask(list(masks_np[f].values())[0])]
                    m, curr_offsets = load_im_into_format_from_image(
                        PIL.Image.fromarray(np.uint8(list(masks_np[f].values())[0])),
                        min_obj_x=xmin, max_obj_x=xmax)

                    masks_attn += [np.asarray(m.resize((attn_size, attn_size)))]
                    masks_dift += [np.asarray(m.resize((dift_size, dift_size)))]
                    masks_relative_size += [np.asarray(m).sum() / (img_size * img_size)]
                    path = f"{dataset_dir}/{vid_id}/{f}.jpg"
                    frame_paths += [path]
                    
                    # Cargar y procesar frame
                    frame = load_im_into_format_from_path(path, offsets=curr_offsets).resize(
                        (img_size, img_size)).convert("RGB")

                masks_relative_size = np.array(masks_relative_size)
                # remove small frames
                if len(np.where(np.array(masks_relative_size) < 0.005)[0]) > 0:
                    # Limpiar memoria antes de continuar
                    #    del masks_np, attn_ls, dift_features
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                query_idx = masks_relative_size.argmax()
                for i in range(3):
                    if i == query_idx:
                        # query
                        #query_dift_features += [dift_features[i]]
                        query_dift_mask += [masks_dift[i]]
                        #query_perdiff_features += [attn_ls[i]]
                        query_labels += [int(vid_id)]
                        query_frames += [frame_paths[i]]
                    else:
                        # gallery
                        #gallery_dift_features += [dift_features[i]]
                        #gallery_perdiff_features += [attn_ls[i]]
                        gallery_masks += [masks_dift[i]]
                        gallery_labels += [int(vid_id)]
                        gallery_frames += [frame_paths[i]]
                
            except torch.cuda.OutOfMemoryError:
                print(f"OOM error processing video {vid_id}, skipping...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            except Exception as e:
                print(f"Error processing video {vid_id}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        query_labels = torch.tensor(query_labels)
        gallery_labels = torch.tensor(gallery_labels)
        
        torch.save([query_frames, query_labels, query_dift_mask, gallery_frames, gallery_labels, gallery_masks], f"{dataset_dir}/ret_data_sansa.pt")
    else:

        """ Retrieval performance on PerMIR """


        datasetDir = "/home/arvc/Marcos/INVESTIGACION/PDM-main/"
        img_size = 518
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        maxScores, maxScore_correct, maxScore_wrong, actualScores, actualScores_correct, actualScores_wrong = [], [], [], [], [], []
        

        query_frames, query_labels, query_dift_mask, \
        gallery_frames, gallery_labels, gallery_masks = torch.load(
            f"{dataset_dir}/ret_data_sansa.pt", weights_only=False)
        topk = 1
        recall_dict = {1: 0, 5: 0, 10: 0, 50: 0}
        ap = []

        for q_idx in tqdm(range(len(query_frames))):
            scores = []

            query_img = Image.open(query_frames[q_idx]).convert('RGB')
            query_img = transform(query_img).unsqueeze(0).unsqueeze(0)  # b t c h w

            support_mask = query_dift_mask[q_idx]  # b t h w
            support_mask = read_mask(support_mask)
            support_mask = F.interpolate(support_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
            support_masks = [support_mask]
            support_masks = torch.stack(support_masks).unsqueeze(0).to(args.device)

            N = 16
            for g_idx in range(0, len(gallery_frames), N):
                start_time = time.time()
                # Limpiar memoria al inicio de cada comparación
                torch.cuda.empty_cache()
                start_time = time.time()
                imgs = query_img
                if g_idx + N > len(gallery_frames):
                    batch_size = len(gallery_frames) - g_idx
                else:
                    batch_size = N
                for b in range(batch_size):
                    gallery_img = Image.open(gallery_frames[g_idx + b]).convert('RGB')
                    #if g_idx == 0 and q_idx in [3,4]:
                    #    gallery_img.save(f"masks/q{q_idx}_g{g_idx + b}_img.jpg")
                    gallery_img = transform(gallery_img).unsqueeze(0).unsqueeze(0)  # b t c h w
                    imgs = torch.cat([imgs, gallery_img], dim=1) # concatenate on time dimension: [1, T, C, H, W]

                #if g_idx == 0 and q_idx in [3,4]:
                #    img1 = Image.open(query_frames[q_idx]).convert('RGB')
                #    img1.save(f"masks/q{q_idx}_query_img.jpg")
                        

                imgs = imgs.to(args.device)
                prompt_dict = build_prompt_dict(support_masks, args.prompt, n_shots=args.shots, train_mode=False, device=model.device)
                

                # object_score_logits[0] es el score del query frame
                # object_score_logits[1:] son los scores de los gallery frames
                # Aplicar sigmoid para convertir logits a probabilidades [0, 1]
                with torch.no_grad():
                    outputs = model(imgs, prompt_dict)
                end_time = time.time()
                #print(f"Time taken for model inference, Index: {g_idx}: {end_time - start_time:.4f} seconds")
                

                pred_masks = outputs["pred_masks"].unsqueeze(0)
                object_score_logits = outputs["object_score_logits"]  # [T, 1] - logits de presencia de objeto
                
                #print(object_score_logits.shape)
                
                # object_score_logits[0] es el score del query frame
                # object_score_logits[1:] son los scores de los gallery frames
                # Aplicar sigmoid para convertir logits a probabilidades [0, 1]
                for b in range(batch_size):
                    gallery_idx = b + 1  # +1 because idx 0 is the query frame
                    if object_score_logits is not None and gallery_idx < len(object_score_logits):
                        gallery_obj_score = object_score_logits[gallery_idx].item()
                        scores.append(gallery_obj_score)
                    else:
                        # Fallback si object_score_logits no está disponible
                        scores.append(0.0)
                
                
                #if g_idx == 0 and q_idx < 8:
                    #mask_support = Image.fromarray(support_mask.cpu().numpy().astype(np.uint8)*255)
                    #mask_support.save(f"masks/q{q_idx}_support.jpg")
                    #print(object_score_logits)
                    #for b in range(batch_size):
                        # pred_masks[0] es el query, pred_masks[1:] son gallery
                        # Así que pred_masks[b+1] corresponde a gallery_frames[g_idx + b]
                        # mask_img1 = convert_mask_values(pred_masks.squeeze(0)[b + 1])
                        # mask_img1.save(f"masks/q{q_idx}_g{g_idx + b}.jpg")


                # Limpiar memoria después de cada comparación
                torch.cuda.empty_cache()
                end_time = time.time()
                #print(f"Time taken for model inference: {end_time - start_time:.4f} seconds")

            pred_scores_idx = torch.argsort(torch.tensor(scores), descending=True)  # change to false fro euclidean
            pred_g_labels = gallery_labels[pred_scores_idx]
            curr_query_lbl = query_labels[q_idx]

            ap += [average_precision_score((gallery_labels == curr_query_lbl).int().numpy(),scores)]
            for r in [1, 5, 10, 50]:
                if curr_query_lbl in pred_g_labels[:r]:
                    recall_dict[r] += 1
            print(f"Query {q_idx+1}/{len(query_frames)}, Predicted class: {pred_g_labels[0]}, Actual class: {curr_query_lbl} - AP: {ap[-1]:.4f}")
            #print(f"Max. score: {torch.max(torch.tensor(scores))}, Scores actual class: {scores[2*q_idx]}, {scores[2*q_idx+1]}")
            if curr_query_lbl == pred_g_labels[0]:
                maxScore_correct.append(torch.max(torch.tensor(scores)).item())
                actualScores_correct.extend([scores[2*q_idx], scores[2*q_idx+1]])
            else:
                maxScore_wrong.append(torch.max(torch.tensor(scores)).item())
                actualScores_wrong.extend([scores[2*q_idx], scores[2*q_idx+1]])
            actualScores.extend([scores[2*q_idx], scores[2*q_idx+1]])
            maxScores.append(torch.max(torch.tensor(scores)).item())
            

        print("MAP:", np.array(ap).mean())
        for k in recall_dict.keys():
            print(f"Recall@{k}", recall_dict[k] / len(query_frames))
        print("Average Retrieved Score (Correct Predictions):", np.mean(maxScore_correct))
        print("Average Retrieved Score (Wrong Predictions):", np.mean(maxScore_wrong))
        print("Average Retrieved Score:", np.mean(maxScores))
        print("Average Score for Actual Class Instances (Correct Predictions):", np.mean(actualScores_correct))
        print("Average Score for Actual Class Instances (Wrong Predictions):", np.mean(actualScores_wrong))
        print("Average Score for Actual Class Instances:", np.mean(actualScores))

    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.name_exp)

    device = "cuda"
    main(args)

