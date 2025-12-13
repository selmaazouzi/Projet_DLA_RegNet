#!/bin/bash

# Vérifie si un chemin de configuration a été fourni
if [ -z "$1" ]; then
    echo "Usage: $0 <chemin_vers_config.yml>"
    echo "Exemple: $0 configs/base_config.yml"
    exit 1
fi

CONFIG_PATH="$1"

# Soumet le job Slurm en appelant train_hpc.sl et en lui passant le chemin de la config
# Le script train_hpc.sl recevra "$CONFIG_PATH" comme argument $1
sbatch train_hpc.sl "$CONFIG_PATH"

echo "Job Slurm soumis pour l'entraînement RegNet avec : ${CONFIG_PATH}"