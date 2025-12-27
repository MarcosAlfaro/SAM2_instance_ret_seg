import torchvision.transforms as transforms



def get_transform(model_name):
    # This function must transform the images depending on the model used.
    # The models will be the DINOv2 variants, SIGLIP base and SAM2 variants.
    if model_name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'siglip-base-patch16-224':
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif model_name in ['sam2_vit_t', 'sam2_vit_b', 'sam2_vit_l']:
        transform = transforms.Compose([
            transforms.Resize(size=(518, 518)),
            #transforms.Resize(size=(1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Model {model_name} not supported for transformations.")
    return transform

    