from .coco import build as build_coco
from .lvis import build as build_lvis
from .fss import build as build_fss
from .deepglobe import build as build_dg
from .isic import build as build_isic
from .lung import build as build_lung
from .pascal_part import build as build_pascal_part
from .paco_part import build as build_paco_part
from .ade20k import build as build_ade20k
from .permis import build as build_permis
from .perseg import build as build_perseg


def build_dataset(dataset_file: str, image_set: str, args=None):
    if dataset_file == 'coco':
        return build_coco(image_set, args)
    if dataset_file == 'lvis':
        return build_lvis(image_set, args)
    if dataset_file == 'fss':
        return build_fss(image_set, args)
    if dataset_file == 'pascal_part':
        return build_pascal_part(image_set, args)
    if dataset_file == 'paco_part':
        return build_paco_part(image_set, args)
    if dataset_file == 'deepglobe':
        return build_dg(image_set, args)
    if dataset_file == 'isic':
        return build_isic(image_set, args)
    if dataset_file == 'lung':
        return build_lung(image_set, args)
    if dataset_file == 'ade20k':
        return build_ade20k(image_set, args)
    if dataset_file == 'permis':
        return build_permis(image_set, args)
    if dataset_file == 'perseg':
        return build_perseg(image_set, args)
    elif dataset_file == 'multi':
        from ds.transform_utils import CustomConcatDataset
        ds_list = [build_dataset(name, image_set="train", args=args) for name in args.multi_train]
        return CustomConcatDataset(dataset_list=ds_list, dataset_ratio=args.ds_weight, samples_per_epoch=210000)
    raise ValueError(f'dataset {dataset_file} not supported')
