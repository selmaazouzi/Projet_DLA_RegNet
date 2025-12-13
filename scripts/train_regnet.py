import argparse
import yaml
from easydict import EasyDict as edict
import torch

# Importation des composants que nous avons définis
from src.models.regnet import RegNet
from src.training.trainer import Trainer
from src.training.dataset import get_data_loaders # <-- Importation du VRAI Data Loader

# La fonction setup_data_loaders (factice) est maintenant remplacée par get_data_loaders.
# Nous allons la renommer ici pour éviter les erreurs d'appel si elle existe encore.

def load_config(config_path):
    """Charge et parse le fichier de configuration YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)

def main():
    """Point d'entrée principal du programme d'entraînement."""
    
    parser = argparse.ArgumentParser(description='Entraînement des modèles RegNet.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Chemin vers le fichier de configuration YAML (e.g., configs/base_config.yml)'
    )
    args = parser.parse_args()
    
    # --- 1. Chargement de la Configuration ---
    cfg = load_config(args.config)
    print(f"Configuration chargée à partir de : {args.config}")
    
    # --- 2. Configuration du Dispositif (Force CPU pour le test) ---
    # Pour le test local, nous forçons l'utilisation du CPU.
    # EN HPC, CECI DEVIENDRAIT : device = torch.device("cuda:0")
    device = torch.device("cpu")
    print(f"ATTENTION : Exécution forcée sur le dispositif : {device}")
    
    # --- 3. Initialisation du Modèle ---
    # Le modèle est créé sur CPU.
    model = RegNet(cfg)
    model = model.to(device)
    # Dans un test réel, il faut s'assurer que RegNet(cfg) n'appelle pas .cuda()
    print(f"Modèle RegNet ({cfg.MODEL.NAME}) initialisé.")
    
    # --- 4. Configuration des DataLoaders ---
    # Pour le test CPU, vous aurez besoin de données ImageNet (même si c'est lent).
    # Si les données ne sont pas présentes, utilisez la version 'factice' ci-dessous
    # ou assurez-vous que cfg.PATHS.DATA_ROOT est correct.
    try:
        train_loader, val_loader = get_data_loaders(cfg)
    except FileNotFoundError:
        print("Erreur : Données ImageNet non trouvées ou chemin incorrect. Utilisation des DataLoaders factices.")
        
        # --- Utilisation des DataLoaders factices pour continuer le test ---
        batch_size = cfg.TRAIN.BATCH_SIZE
        image_size = 224
        
        dummy_input = torch.randn(batch_size, 3, image_size, image_size)
        dummy_target = torch.randint(0, 1000, (batch_size,))
        dummy_dataset = torch.utils.data.TensorDataset(dummy_input, dummy_target)
        
        train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 sur CPU
        val_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        # -------------------------------------------------------------------
        
    # --- 5. Initialisation du Trainer ---
    # Pour le test, nous allons surcharger l'époque pour qu'il s'exécute rapidement.
    cfg.TRAIN.EPOCHS = 3 # Réduit à 3 époques pour la validation rapide
    
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device # <-- PASSAGE DU DISPOSITIF (essentiel pour CPU)
    )
    
    # --- 6. Démarrage de l'entraînement (Boucle d'entraînement complète) ---
    print("\n--- Démarrage de la simulation d'entraînement (3 époques) ---")
    
    # Nous appelons la méthode train() complète qui gère les époques, validate(), et save_checkpoint().
    trainer.train() 
        
    print("--- Simulation d'entraînement terminée ---")
    
if __name__ == '__main__':
    # La fonction main nécessite que get_data_loaders soit définie.
    # Si vous n'avez pas encore implémenté src/training/dataset.py, le test utilisera les DataLoaders factices.
    main()