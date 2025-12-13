#!/bin/bash
#
# Fichier de soumission Slurm pour l'entraînement de RegNet
#

# --- DIRECTIVES SLURM ---

# Partition de travail (Adapter selon votre environnement HPC)
#SBATCH -p mesonet 

# Nombre de nœuds
#SBATCH -N 1 

# Nombre de tâches (nous utilisons 1 tâche pour l'entraînement)
#SBATCH -n 1

# Nombre de CPUs par tâche (Doit correspondre à NUM_WORKERS dans YAML)
#SBATCH -c 8                          

# Ressources GPU (1 GPU par défaut, ajustez si multi-GPU est requis)
#SBATCH --gres=gpu:1                  

# Temps maximum alloué pour le job (Ajustez selon la durée estimée de 100 époques)
#SBATCH --time=24:00:00               

# Mémoire par nœud
#SBATCH --mem=32G                     

# Nom du compte (Remplacer par votre ID de projet/compte HPC)
#SBATCH --account=M25206              

# Fichiers de sortie/erreur (Le %j sera l'ID du job Slurm)
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

# --- COMMANDES D'EXÉCUTION ---

# Vérifie si le chemin de configuration est passé en argument
if [ -z "$1" ]; then
    echo "Erreur : Veuillez spécifier le chemin vers le fichier de configuration YAML."
    exit 1
fi

CONFIG_PATH="$1"

# 1. Chargement de l'environnement virtuel (Ajustez le chemin si nécessaire)
source ~/env_dla/bin/activate

# 2. Lancement du script principal
echo "Lancement de l'entraînement avec configuration : ${CONFIG_PATH}"
python scripts/train_regnet.py --config "${CONFIG_PATH}"