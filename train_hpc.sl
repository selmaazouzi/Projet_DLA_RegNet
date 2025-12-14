#!/bin/bash
#
# Fichier de soumission Slurm pour l'entraînement de RegNet
#

# --- DIRECTIVES SLURM ---

# Nom du job (Pour faciliter le suivi dans squeue)
#SBATCH --job-name=MiniImageNet_Train

# Partition de travail (mesonet est la partition à privilégier pour Juliet [cite: 124])
#SBATCH -p mesonet

# Nombre de nœuds (1 est suffisant pour le moment [cite: 139])
#SBATCH -N 1

# Nombre de tâches (nous utilisons 1 tâche pour l'entraînement)
#SBATCH -n 1

# Nombre de CPUs par tâche (8 est cohérent avec NUM_WORKERS: 8 [cite: 151])
#SBATCH -c 8

# Ressources GPU (1 GPU par défaut [cite: 161, 168])
#SBATCH --gres=gpu:1

# TEMPS MAXIMUM RÉDUIT POUR LE TEST (Formatage strict HH:MM:SS)
#SBATCH --time=00:10:00

# Mémoire par nœud (Réduit à 8G pour la soumission de test)
#SBATCH --mem=8G

# Nom du compte (M25206 [cite: 55, 202])
#SBATCH --account=M25206

# Fichiers de sortie/erreur (Les dossiers logs/ doivent exister)
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

# --- COMMANDES D'EXÉCUTION ---

# Vérifie si le chemin de configuration est passé en argument
if [ -z "$1" ]; then
    echo "Erreur : Veuillez spécifier le chemin vers le fichier de configuration YAML."
    exit 1
fi

CONFIG_PATH="$1"

# 1. Chargement de l'environnement virtuel [cite: 254]
source ~/Projet_DLA_RegNet/env_dla/bin/activate

# 2. Déplacement vers le répertoire du projet (pour la résolution des chemins)
cd ~/Projet_DLA_RegNet/
export PYTHONPATH=$HOME/Projet_DLA_RegNet:$PYTHONPATH
# 3. Lancement du script principal
echo "Lancement de l'entraînement avec configuration : ${CONFIG_PATH}"
python scripts/train_regnet.py --config "${CONFIG_PATH}"

# 4. Désactivation de l'environnement (Bonne pratique)
deactivate
