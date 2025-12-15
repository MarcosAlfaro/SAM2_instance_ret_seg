import os
from PIL import Image
from typing import Any, Dict, List, Tuple

import py3_wget
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
import torch.nn.functional as F
import numpy as np

from models.sam2.modeling.sam2_utils import preprocess
from models.sam2.modeling.sam2_base import SAM2Base 
from models.sansa.model_utils import BackboneOutput, DecoderOutput
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL
from util.promptable_utils import rescale_prompt
import time

def convert_mask_values(mask):
        # If mask has multiple channels [3, H, W] or [1, H, W], take the first one/squeeze
        if mask.dim() == 3:
            mask = mask[0]
        mask = mask.cpu().detach().numpy()
        mask[mask >= 0.0] = 255
        mask[mask < 0.0] = 0
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        return mask




class SANSA(nn.Module):
    def __init__(self, sam: SAM2Base, device: torch.device):
        super().__init__()
        self.sam = sam
        self.device = device
    
    def masked_avg_pool(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masked average pooling over spatial dimensions.
        
        Args:
            features: [B, C, H, W] feature map from two-way transformer
            mask: [B, 1, H', W'] predicted mask (will be resized to match features)
        
        Returns:
            pooled: [B, C] masked average pooled features
        """
        # Resize mask to match feature spatial dimensions
        B, C, H, W = features.shape
        mask_resized = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        
        # Apply sigmoid to convert logits to probabilities [0, 1]
        mask_probs = torch.sigmoid(mask_resized)  # [B, 1, H, W]

        # transform mask_probs to binary mask (0 or 1) using threshold 0
        #mask_probs = (mask_probs >= 0.5).float()
        
        # Weighted sum: sum over spatial dims weighted by mask
        weighted_sum = (features * mask_probs).sum(dim=[2, 3])  # [B, C]
        
        # Normalize by sum of mask weights to get average
        mask_sum = mask_probs.sum(dim=[2, 3]) + 1e-8  # [B, 1] + epsilon for stability
        
        return weighted_sum / mask_sum  # [B, C] 

    def forward(self, samples: torch.Tensor, prompt_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run SANSA.
        Args:
            samples: [B, T, C, H, W].
            targets: list (len B) of dicts with:
                - 'is_support': list[bool] of len T
                - 'masks': Tensor [T, H, W]

        Returns:
            {"pred_masks": Tensor [B*T, H', W']}
        """
        start_time_1 = time.time()
        samples, B, T, orig_size = self._preprocess_visual_features(samples, self.sam.image_size)
        end_time_1 = time.time()
        #print(f"Preprocessing time: {end_time_1 - start_time_1:.4f} seconds")
        start_time_2 = time.time()
        backbone_output: BackboneOutput = self._forward_backbone(samples, orig_size)
        end_time_2 = time.time()
        #print(f"Backbone forward time: {end_time_2 - start_time_2:.4f} seconds")
        outputs = {"masks": [], "object_score_logits": [], "memory_embeddings": []}

        start_time_3 = time.time()
        n_shots = prompt_dict['shots']
        for b in range(B):
            self.memory_bank = {}
            self.memory_bank_idx0 = {}
            for idx in range(T):
                absolute_idx = b * T + idx

                if idx < n_shots:
                    frame_prompt = prompt_dict[b][idx]['prompt']
                    prompt_type = prompt_dict[b][idx]['prompt_type']
                    frame_prompt = rescale_prompt(frame_prompt, prompt_type, orig_size[b], self.sam.image_size)
                    if prompt_type == 'mask':
                        decoder_out: DecoderOutput = self.sam._use_mask_as_output(backbone_output, frame_prompt, absolute_idx)
                    else:
                        decoder_out: DecoderOutput = self._compute_decoder_out_no_mem(backbone_output, absolute_idx, prompt_input=frame_prompt)
                        
                else:
                    start_time_5 = time.time()
                    # CRITICAL FIX: Always use memory_idx=1 for all gallery frames (idx >= n_shots)
                    # This prevents the memory mechanism from thinking it's processing a long sequence
                    # All gallery frames should behave as "frame 1" in a 2-frame sequence (query + gallery)
                    fixed_memory_idx = 1
                    decoder_out: DecoderOutput = self._compute_decoder_out_w_mem(backbone_output, absolute_idx, fixed_memory_idx, self.memory_bank_idx0)
                    end_time_5 = time.time()
                    #print(f"Decoder with memory time, idx={idx}: {end_time_5 - start_time_5:.4f} seconds")
                
                # Ensure we only have one mask per frame (the first one)
                # This is crucial because _use_mask_as_output (used for query) returns 3 masks by default
                if decoder_out.masks.shape[1] > 1:
                    decoder_out.masks = decoder_out.masks[:, 0:1]

                # update memory bank - but only for the query frame (idx 0)
                # Don't update for gallery frames to avoid memory accumulation
                if idx < n_shots:
                    mem_entry = self._compute_memory_bank_dict(decoder_out, backbone_output, absolute_idx)
                    if idx == 0:
                        self.memory_bank_idx0[0] = mem_entry
                        # Store the query frame's memory embedding separately (won't affect gallery processing)
                
                outputs["masks"].append(decoder_out.masks[0])
                mask_img = convert_mask_values(decoder_out.masks[0])

                mask_img.save(f"masks/sansa_mask_b{b}_idx{idx}.png")
                
                # Apply masked average pooling if we have the transformer features
                if decoder_out.pix_feat_with_mem is not None and decoder_out.masks is not None:
                    # decoder_out.masks has shape [B, 1, H, W] (batch, 1 best mask, height, width)
                    # Take only the first mask [H, W] and reshape to [1, 1, H, W]
                    first_mask = decoder_out.masks[0, 0]  # [H, W] - first batch, first mask
                    mask_for_pooling = first_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    pooled_embedding = self.masked_avg_pool(decoder_out.pix_feat_with_mem, mask_for_pooling)
                    decoder_out.mask_pooled_features = pooled_embedding  # [1, C]
                    if "mask_pooled_embeddings" not in outputs:
                        outputs["mask_pooled_embeddings"] = []
                    outputs["mask_pooled_embeddings"].append(pooled_embedding)
                
                # Store object pointer for ALL frames (both query and gallery)
                # obj_ptr is a compact embedding [1, C, H, W] representing the object
                if decoder_out.obj_ptr is not None:
                    outputs["memory_embeddings"].append(decoder_out.obj_ptr)
                
                # Collect object score logits (probability of object presence)
                if decoder_out.object_score_logits is not None:
                    outputs["object_score_logits"].append(decoder_out.object_score_logits)

        masks = torch.cat(outputs["masks"])
        masks = F.interpolate(masks[None], size=orig_size[0], mode='bilinear', align_corners=False)[0]
        end_time_3 = time.time()
        #print(f"Decoding time: {end_time_3 - start_time_3:.4f} seconds")
        start_time_4 = time.time()
        object_score_logits = None
        if len(outputs["object_score_logits"]) > 0:
            object_score_logits = torch.cat(outputs["object_score_logits"], dim=0)  # [T, 1]
        
        # Concatenate object pointers for all frames: [B*T, C, H, W]
        obj_pointers = None
        if len(outputs["memory_embeddings"]) > 0:
            obj_pointers = torch.cat(outputs["memory_embeddings"], dim=0)  # [B*T, C, H, W]
        
        # Concatenate masked pooled embeddings: [B*T, C]
        mask_pooled_embeddings = None
        if "mask_pooled_embeddings" in outputs and len(outputs["mask_pooled_embeddings"]) > 0:
            mask_pooled_embeddings = torch.cat(outputs["mask_pooled_embeddings"], dim=0)  # [B*T, C]
        
        return {
            "pred_masks": masks, 
            "object_score_logits": object_score_logits, 
            "obj_pointers": obj_pointers,
            "mask_pooled_embeddings": mask_pooled_embeddings
        }

    def _preprocess_visual_features(
        self, samples: torch.Tensor, image_size: int
    ) -> Tuple[torch.Tensor, int, int, List[Tuple[int, int]]]:
        """
        Flatten [B,T,C,H,W] -> [B*T,C,H,W], store original sizes, and apply SAM2 preprocess.

        Args:
            samples:   Tensor [B, T, C, H, W].
            image_size: target side for SAM2 preprocessing.

        Returns:
            (samples_bt, B, T, orig_sizes)
        """

        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        return samples, B, T, orig_size

    def _compute_decoder_out_no_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        prompt_input: Dict[str, torch.Tensor] | None,
    ) -> DecoderOutput:
        """
        Decode a frame without memory: used for reference frames;

        Args:
            backbone_out: backbone features.
            idx: absolute idx.
            prompt:       "mask" | "point" | "scribble" | "box".
            prompt_input: inputs for point/scribble/box (ignored for "mask").

        Returns:
            DecoderOutput.
        """
        current_vision_feats = backbone_out.get_current_feats(idx)

        high_res_features = backbone_out.get_high_res_features(current_vision_feats)

        pix_feat_no_mem = current_vision_feats[-1:][-1] + self.sam.no_mem_embed
        pix_feat_no_mem = pix_feat_no_mem.permute(1, 2, 0).view(1, 256, 64, 64)
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_no_mem,
            point_inputs=prompt_input,
            high_res_features=high_res_features,
            multimask_output=False,  # Disable multiple mask outputs
        )
        return decoder_out

    def _compute_decoder_out_w_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        memory_idx: int,
        memory_bank: Dict[int, Dict[str, torch.Tensor]],
    ) -> DecoderOutput:
        """
        Decode a frame with memory: used for target frames;

        Args:
            backbone_out: backbone features.
            idx: absolute idx.
            memory_idx:   temporal index t (0-based).
            memory_bank:  dict of memory entries from previous frames.

        Returns:
            DecoderOutput
        """
        current_vision_feats = backbone_out.get_current_feats(idx)
        current_vision_pos_embeds = backbone_out.get_current_pos_embeds(idx)

        # take only the highest res feature map
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)
        
        pix_feat_with_mem = self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],
            num_frames=memory_idx+1,
            memory_bank=memory_bank
        )

        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            high_res_features=high_res_features,
            multimask_output=False,  # Always generate 1 mask
        )
        return decoder_out

    def _compute_memory_bank_dict(
        self, decoder_out: DecoderOutput, backbone_out: BackboneOutput, idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Encode current prediction into memory for later frames.

        Args:
            decoder_out: decoder output with high_res/low_res masks.
            backbone_out:  backbone features.
            idx: absolute idx.

        Returns:
            Memory entry dict.
        """
        current_vision_feats = backbone_out.get_current_feats(idx)
        feat_sizes = backbone_out.feat_sizes

        mem_feats, mem_pos = self.sam._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )
        return {
            "maskmem_features": mem_feats,
            "maskmem_pos_enc": mem_pos,
            "pred_masks": decoder_out.low_res_masks,
            "obj_ptr": decoder_out.obj_ptr,
        }

    def _forward_backbone(
        self, samples: torch.Tensor, orig_size: List[Tuple[int, int]]
    ) -> BackboneOutput:
        """
        Run SAM2 image encoder and prepare backbone features for decoding.

        Args:
            samples:  Tensor [B*T, C, H, W] after preprocessing.
            orig_size:   list of original frame sizes.

        Returns:
            BackboneOutput.
        """
        vis = self.sam.image_encoder.trunk(samples)
        feats, pos = self.sam.image_encoder.neck(vis)

        # discard lowest resolution
        feats, pos = feats[:-1], pos[:-1]

        feats[0] = self.sam.sam_mask_decoder.conv_s0(feats[0])
        feats[1] = self.sam.sam_mask_decoder.conv_s1(feats[1])

        bb = {
            "vision_features": feats[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": feats,
        }
        vision_feats, vision_pos, sizes = self.sam._prepare_backbone_features(bb)
        return BackboneOutput(orig_size, vision_feats, vision_pos, sizes)


def build_sansa(sam2_version: str = 'large', adaptformer_stages: List[int] = [2, 3], channel_factor: float = 0.3, device: str = 'cuda',
                obj_pred_scores: bool = False) -> SANSA:
    assert sam2_version in SAM2_PATHS_CONFIG.keys(), f'wrong argument sam2_version: {sam2_version}'
    
    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[sam2_version]
    if not os.path.isfile(sam2_weights):
        print(f"Downloading SAM2-{sam2_version}")
        py3_wget.download_file(SAM2_WEIGHTS_URL[sam2_version], sam2_weights)

    with initialize(version_base=None, config_path=".", job_name="test_app"):
        cfg = compose(config_name=sam2_config, overrides=[
            f"++model.image_encoder.trunk.adaptformer_stages={adaptformer_stages}",
            f"++model.image_encoder.trunk.adapt_dim={channel_factor}",
        ])

        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores = obj_pred_scores
        cfg.model.pred_obj_scores_mlp = obj_pred_scores
        cfg.model.fixed_no_obj_ptr = obj_pred_scores
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location="cpu", weights_only=False)["model"]
    sam.load_state_dict(state_dict, strict=False)
    model = SANSA(sam=sam, device=torch.device(device))

    # freeze everything except adapters
    for name, p in model.named_parameters():
        p.requires_grad = ("adapter" in name)

    return model
