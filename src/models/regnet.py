import numpy as np
import math
from typing import List, Tuple
import torch.nn as nn
# Importation des blocs validés depuis le package 'blocks'
from src.blocks import Stem, Head, Stage 


# -----------------------------------------------------------------------------
# 1. Fonction de Paramétrisation Linéaire Quantifiée (Cœur RegNet)
# -----------------------------------------------------------------------------

def generate_regnet_parameters(d: int, w0: float, wa: float, wm: float) -> Tuple[List[int], List[int]]:
    """
    Calcule les largeurs (w_i) et les profondeurs (d_i) par étage (stage) pour un RegNet.
    (Implémentation des Équations 2, 3 et 4 de l'article)
    
    Args:
        d (int): Profondeur totale du réseau (nombre de blocs).
        w0 (float): Largeur initiale (w_0).
        wa (float): Pente linéaire (w_a).
        wm (float): Multiplicateur de largeur pour la quantification (w_m).
        
    Retourne:
        Tuple[List[int], List[int]]: (largeurs_stages, profondeurs_stages)
    """
    
    # Étape 1 : Progression linéaire théorique u_j = w0 + wa * j (Eq. 2)
    block_indices = np.arange(d)
    u_j_continuous = w0 + wa * block_indices 

    # Étape 2 : Calcul et arrondi de l'indice de quantification s_j (Eq. 3)
    s_j_raw = np.log(u_j_continuous / w0) / np.log(wm)
    s_j_quantized = np.round(s_j_raw).astype(int)
    
    # Étape 3 : Regroupement et Quantification finale (Eq. 4)
    unique_widths = []
    profondeurs_stages = []
    
    # Trouver les indices de stage uniques pour itérer (ex: 0, 1, 2, 3)
    stage_indices = np.unique(s_j_quantized)
    
    for stage_index in stage_indices:
        # Profondeur d_i: nombre de blocs ayant le même indice quantifié
        depth_i = np.sum(s_j_quantized == stage_index)
        
        # Largeur w_i théorique: w_0 * w_m^stage_index
        width_i_raw = w0 * (wm ** stage_index)
        
        # [cite_start]Rendre la largeur divisible par 8 (contrainte d'efficacité GPU, [cite: 208])
        # math.ceil(x / 8.0) * 8 est la manière standard d'arrondir à la puissance de 8 supérieure.
        width_i = math.ceil(width_i_raw / 8.0) * 8
        
        if depth_i > 0:
            unique_widths.append(width_i)
            profondeurs_stages.append(depth_i)
            
    return unique_widths, profondeurs_stages


# -----------------------------------------------------------------------------
# 2. Assembleur RegNet (Lecture de la configuration YAML)
# -----------------------------------------------------------------------------

class RegNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Extraction des 6 paramètres structurels du design space
        d = cfg.MODEL.NETWORK_DEPTH
        w0 = cfg.MODEL.INITIAL_WIDTH
        wa = cfg.MODEL.SLOPE
        wm = cfg.MODEL.QUANTIZED_PARAM
        b = cfg.MODEL.BOTTLENECK_RATIO
        g = cfg.MODEL.GROUP_WIDTH
        
        # 1. Calculer la structure (largeurs et profondeurs par stage)
        largeurs_stages, profondeurs_stages = generate_regnet_parameters(d, w0, wa, wm)
        
        # --- Construction ---
        
        # [cite_start]1. Stem (Tige) - Fixé à 32 canaux [cite: 179]
        STEM_WIDTH = 32 
        self.stem = Stem(in_channels=3, out_channels=STEM_WIDTH)
        current_in_width = STEM_WIDTH
        
        # 2. Body (Corps: Stages)
        self.body = nn.Sequential()
        
        for i in range(len(largeurs_stages)):
            stage_depth = profondeurs_stages[i]
            stage_width = largeurs_stages[i]
            
            # Logique de Stride : Le Stem est stride=2. 
            # Stage 1 (i=0) est stride=1 (conserve la résolution),
            # Stages 2, 3, 4 (i>0) commencent par stride=2 (réduisent la résolution).
            stride_first_block = 1 if i == 0 else 2

            stage_module = Stage(
                in_channels=current_in_width, 
                out_channels=stage_width, 
                depth=stage_depth, 
                bottleneck_ratio=b, 
                group_width=g,
                stride_first_block=stride_first_block 
            )
            
            self.body.add_module(f'stage{i+1}', stage_module)
            current_in_width = stage_width
        
        # 3. Head (Tête)
        self.head = Head(in_channels=current_in_width, num_classes=1000)

        # Print pour vérification (utile lors de l'exécution du script de lancement)
        print(f"Structure RegNet Générée avec {len(largeurs_stages)} Stages:")
        for i in range(len(largeurs_stages)):
             print(f"Stage {i+1}: Largeur={largeurs_stages[i]}, Profondeur={profondeurs_stages[i]}")

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x