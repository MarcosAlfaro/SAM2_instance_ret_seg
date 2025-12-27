import torch
from transformers import AutoProcessor, AutoModel


def load_model(model_name='dinov2_vits14', device='cuda'):
    """
    Load the model to extract the top-K candidates for instance retrieval.
    
    Args:
        model_name: Name of the model to load
        device: "cuda" or "cpu"
    
    Returns:
        tuple: (model, processor), where processor is None for DinoV2 models
    """
    print(f"Loading model {model_name}")
    
    if model_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        return model, None
    elif model_name == 'siglip-base-patch16-224':
        model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224",
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        return model, processor
    else:
        raise ValueError(f"Model {model_name} not supported.")
    