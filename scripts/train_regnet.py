import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import mlflow 

# Import de vos modules
from src.models import RegNet, create_efficientnet # <-- Utilise le créateur EfficientNet
from src.training import Trainer, get_data_loaders
# from src.blocks import Stem, Head # Non nécessaire ici

# --- Configuration du Log (doit correspondre à votre main.py) ---
class Config:
    """ Permet d'accéder aux paramètres du YAML via point (cfg.TRAIN.EPOCHS). """
    def __init__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

# --- FONCTION DE SETUP MLFLOW ---
def setup_mlflow(cfg):
    """ Configure l'expérience MLflow et enregistre les hyper-paramètres. """
    
    # Définir le nom de l'expérience
    experiment_name = f"Classification_{cfg.MODEL.NAME}"
    mlflow.set_experiment(experiment_name)
    
    # Enregistrer les hyper-paramètres
    print(f"Démarrage de l'expérience MLflow : {experiment_name}")
    
    # Log les paramètres du modèle
    for key, value in vars(cfg.MODEL).items():
        mlflow.log_param(f"model.{key}", value)
    
    # Log les paramètres d'entraînement
    for key, value in vars(cfg.TRAIN).items():
        mlflow.log_param(f"train.{key}", value)

def main(cfg):
    
    # Définition du nom du Run MLflow plus adapté à RegNet ou EfficientNet
    if 'REGNET' in cfg.MODEL.NAME.upper():
        run_name = f"{cfg.MODEL.NAME}_D{cfg.MODEL.NETWORK_DEPTH}_LR{cfg.TRAIN.LR}"
    elif 'EFFICIENTNET' in cfg.MODEL.NAME.upper():
        run_name = f"{cfg.MODEL.NAME}_{cfg.MODEL.VARIANT}_LR{cfg.TRAIN.LR}"
    else:
        run_name = f"{cfg.MODEL.NAME}_LR{cfg.TRAIN.LR}"
    
    with mlflow.start_run(run_name=run_name):
        
        # 1. Enregistrement des paramètres MLflow
        setup_mlflow(cfg)

        # 2. Définition du périphérique (DEVICE)
        device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du périphérique : {device}")

        # 3. Chargement des données
        train_loader, val_loader = get_data_loaders(cfg)

        # --- 4. Initialisation du modèle (LOGIQUE DE SÉLECTION) ---
        if 'REGNET' in cfg.MODEL.NAME.upper():
            # Utilise l'assembleur RegNet
            model = RegNet(cfg)
        elif 'EFFICIENTNET' in cfg.MODEL.NAME.upper():
            # Utilise l'assembleur EfficientNet
            model = create_efficientnet(cfg)
        else:
             raise ValueError(f"Nom de modèle non reconnu: {cfg.MODEL.NAME}. Doit contenir 'RegNet' ou 'EfficientNet'.")
             
        model = model.to(device)
        
        # --- SUPPORT MULTI-GPU (DataParallel) ---
        if torch.cuda.device_count() > 1:
            print(f"Utilisation de {torch.cuda.device_count()} GPUs via DataParallel.")
            model = torch.nn.DataParallel(model) 
        # ----------------------------------------
        
        print(f"Modèle {cfg.MODEL.NAME} initialisé.")

        # 5. Définition de la Loss et de l'Optimiseur
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), 
                              lr=cfg.TRAIN.LR, 
                              momentum=cfg.TRAIN.MOMENTUM, 
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        # 6. Démarrage de l'entraînement
        trainer = Trainer(model, criterion, optimizer, device, cfg)
        trainer.train(train_loader, val_loader, cfg.TRAIN.EPOCHS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RegNet/EfficientNet Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # Lecture du fichier YAML
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        
    # Lancement de l'exécution avec l'objet Config
    main(Config(config_data))