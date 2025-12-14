import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

# --- Constantes pour la Normalisation ---

# Valeurs moyennes et écarts-types standard d'ImageNet (RVB)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Statistiques standards de CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# --- Transformations Spécifiques aux Datasets ---

def get_imagenet_transformations(is_train: bool) -> transforms.Compose:
    """ Définit les transformations de données standard pour ImageNet/Mini-ImageNet (224x224). """
    
    if is_train:
        # Transformations pour l'entraînement (inclut l'augmentation des données)
        return transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Transformations pour la validation/test (découpage central déterministe)
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

def get_cifar10_transformations(is_train: bool) -> transforms.Compose:
    """ Définit les transformations pour CIFAR-10 (32x32). """
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_data_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Charge les DataLoaders, basculant entre ImageNet-style (Mini-ImageNet) et CIFAR-10
    en fonction du chemin spécifié dans la configuration.
    """
    
    data_root = cfg.PATHS.DATA_ROOT
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.TRAIN.NUM_WORKERS
    
    # Détection automatique du dataset basé sur le chemin DATA_ROOT
    is_cifar = 'cifar' in data_root.lower()
    
    # --- 1. Choix et Chargement des Datasets ---
    
    if is_cifar:
        print(f"Chargement du dataset CIFAR-10 (téléchargement automatique vers {data_root}).")
        transform_train = get_cifar10_transformations(is_train=True)
        transform_val = get_cifar10_transformations(is_train=False)

        train_dataset = datasets.CIFAR10(
            root=data_root, 
            train=True, 
            download=True, 
            transform=transform_train
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False, 
            download=True, 
            transform=transform_val
        )
        dataset_name = "CIFAR-10"

    else:
        print(f"Chargement du dataset ImageNet-style (Mini-ImageNet) depuis {data_root}.")
        
        # Le Mini-ImageNet DOIT avoir la structure data_root/train/classe_id
        train_dataset = datasets.ImageFolder(
            root=f"{data_root}/train",
            transform=get_imagenet_transformations(is_train=True)
        )
        val_dataset = datasets.ImageFolder(
            root=f"{data_root}/val",
            transform=get_imagenet_transformations(is_train=False)
        )
        dataset_name = "Mini-ImageNet (ImageFolder)"

    # --- 2. Création des DataLoaders ---
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"DataLoaders {dataset_name} configurés. Entraînement: {len(train_loader)} batchs. Validation: {len(val_loader)} batchs.")
    return train_loader, val_loader