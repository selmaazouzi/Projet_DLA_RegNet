# 🧠 RegNet_DLA_Project : Conception de l'Espace de Design Réseau (RegNet)

Ce projet implémente la méthodologie **"Designing Network Design Spaces" (RegNet)** proposée par **Facebook AI Research (FAIR)**, en se concentrant sur la **paramétrisation linéaire quantifiée** des architectures de réseaux neuronaux.

L'objectif principal est de **reproduire et analyser l'architecture RegNetX**, dans un cadre expérimental orienté **Deep Learning Avancé**, avec un entraînement sur le jeu de données **ImageNet**.

---

## 📂 Structure du Projet

```text
RegNet_DLA_Project/
├── configs/            # Fichiers de configuration YAML pour les expériences
├── data/               # Données locales (ignorées par Git)
├── scripts/            # Scripts de lancement (local & HPC)
├── src/                # Code source principal
│   ├── blocks/         # Blocs de base (Stem, Head, XBlock, Stage)
│   ├── models/         # Logique RegNet (paramétrisation & construction)
│   └── training/       # Entraînement (Trainer, DataLoaders, Scheduler)
├── logs/               # Logs d'exécution (Slurm)
├── saved_models/       # Checkpoints des modèles
├── requirements.txt    # Dépendances Python
└── train_hpc.sl        # Script de soumission Slurm
```

---

## 🛠️ Installation de l'Environnement

### 1. Création de l'environnement virtuel

```bash
python -m venv env_dla
source env_dla/bin/activate
```

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

Les dépendances principales incluent :
- PyTorch
- TorchVision
- NumPy
- PyYAML
- EasyDict

---

## 🚀 Lancement des Expériences (HPC – Slurm)

Ce projet est conçu pour une exécution sur infrastructure **HPC** via **Slurm**.

### 1. Configuration des hyperparamètres

Les expériences sont définies via des fichiers YAML :

- `configs/base_config.yml`  
  → Configuration RegNetX par défaut (taille intermédiaire)

- `configs/exp_bottleneck.yml`  
  → Variante avec **bottleneck ratio b = 4**

⚠️ **Action requise** : mettre à jour le chemin vers ImageNet dans `configs/base_config.yml`

```yaml
PATHS:
  DATA_ROOT: '/chemin/vers/data/imagenet_dossier_racine'
```

### 2. Soumission du job Slurm

Utiliser le script de lancement :

```bash
# Expérience de base
./scripts/launch_hpc.sh configs/base_config.yml

# Expérience bottleneck b=4
./scripts/launch_hpc.sh configs/exp_bottleneck.yml
```

### 3. Suivi des jobs

```bash
squeue -u $USER
```

Les logs sont générés dans le dossier `logs/` :

- `slurm-<JOB_ID>.out`
- `slurm-<JOB_ID>.err`

---

## 📊 Objectifs du Projet

- Implémentation fidèle de la **paramétrisation RegNet**
- Étude de l'impact des hyperparamètres structuraux
- Comparaison des variantes RegNetX
- Entraînement et évaluation sur ImageNet
- Approche modulaire et reproductible

---

## 📚 Références

- Radosavovic et al., *Designing Network Design Spaces*, FAIR
- Facebook AI Research – RegNet

---

📌 *Projet réalisé dans le cadre du module Deep Learning Avancé.*
