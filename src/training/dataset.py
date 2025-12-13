import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

def get_imagenet_transformations(is_train: bool) -> transforms.Compose:
    """
    Définit les transformations de données standard pour ImageNet.
    
    Arguments:
        is_train (bool): True pour l'ensemble d'entraînement (avec augmentation), 
                         False pour la validation/test.
                         
    Retourne:
        transforms.Compose: La séquence de transformations à appliquer.
    """
    
    # Valeurs moyennes et écarts-types standard d'ImageNet (RVB)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    if is_train:
        # Transformations pour l'entraînement (inclut l'augmentation des données)
        return transforms.Compose([
            # Redimensionnement aléatoire et découpage (RandomResizedCrop)
            # Cette étape est cruciale pour l'augmentation des données.
            transforms.RandomResizedCrop(224), 
            # Retournement horizontal aléatoire
            transforms.RandomHorizontalFlip(),
            # Conversion en tenseur PyTorch
            transforms.ToTensor(),
            # Normalisation
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Transformations pour la validation/test (découpage central déterministe)
        return transforms.Compose([
            # Redimensionnement pour s'assurer que le côté court est de 256
            transforms.Resize(256),
            # Découpage central pour obtenir une image 224x224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalisation
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_data_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Charge les DataLoaders pour les ensembles d'entraînement et de validation d'ImageNet.
    
    Arguments:
        cfg (EasyDict): Objet de configuration chargé à partir du YAML.
        
    Retourne:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    
    data_root = cfg.PATHS.DATA_ROOT # Chemin vers le répertoire ImageNet (e.g., /chemin/vers/data/imagenet)
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.TRAIN.NUM_WORKERS
    
    # --- 1. Datasets ---
    
    # Ensemble d'entraînement (avec augmentation)
    train_dataset = datasets.ImageFolder(
        root=f"{data_root}/train", # On suppose que les données sont dans data_root/train et data_root/val
        transform=get_imagenet_transformations(is_train=True)
    )
    
    # Ensemble de validation (sans augmentation)
    val_dataset = datasets.ImageFolder(
        root=f"{data_root}/val",
        transform=get_imagenet_transformations(is_train=False)
    )
    
    # --- 2. DataLoaders ---
    
    # DataLoader d'entraînement
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Important pour l'entraînement
        num_workers=num_workers, # Pour le chargement parallèle des données (très important en HPC)
        pin_memory=True # Optimisation pour les GPU
    )
    
    # DataLoader de validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # Pas nécessaire de mélanger pour la validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"DataLoaders ImageNet configurés. Entraînement: {len(train_loader)} batchs. Validation: {len(val_loader)} batchs.")
    return train_loader, val_loader

# Note: Ce code nécessite que le répertoire data_root contienne les sous-dossiers 'train' et 'val' 
# avec la structure de classification PyTorch (dossiers par classe).