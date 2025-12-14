import torch.nn as nn
import torch 
import math
from typing import Optional

# -----------------------------------------------------------------------------
# Bloc SE (Squeeze-and-Excitation) pour RegNetY
# -----------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Bloc Squeeze-and-Excitation (SE) standard, utilisé dans RegNetY.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        
        # Le canal intermédiaire est défini par le rapport de réduction (généralement 4 pour RegNet)
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        # Squeeze : Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation : Deux couches Fully Connected (FC)
        # Couche 1 : Réduction
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # Couche 2 : Expansion
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        w = self.avg_pool(x)
        
        # Excitation
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        
        # Mise à l'échelle (Scale)
        return x * w # Multiplie la carte de caractéristiques d'entrée par le poids appris (w)


# -----------------------------------------------------------------------------
# Bloc Résiduel Bottleneck (XBlock) avec option SE
# -----------------------------------------------------------------------------

class XBlock(nn.Module):
    """
    Bloc résiduel Bottleneck avec Group Convolution (X Block).
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, bottleneck_ratio: int, group_width: int, stride: int = 1, use_se: bool = False):
        super().__init__()
        
        self.use_se = use_se
        
        # Le nombre de canaux à l'intérieur du bottleneck est out_channels / bottleneck_ratio
        bottleneck_channels = out_channels // bottleneck_ratio
        
        # Calcul du nombre de groupes (num_groups) pour la conv 3x3
        num_groups = bottleneck_channels // group_width
        
        # --- Vérification de la Divisibilité (Robustesse) ---
        if bottleneck_channels % group_width != 0:
            # Cette vérification est essentielle pour garantir que la division est entière.
            # Normalement, RegNet assure cette divisibilité. Si ce n'est pas le cas, 
            # nous devons ajuster num_groups.
            num_groups = max(1, math.floor(bottleneck_channels / group_width))

        # Shortcut (Branche latérale résiduelle)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 1. Première Conv 1x1 (Réduction)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 2. Conv 3x3 Groupée (Calcul spatial)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=num_groups, # Groupes basés sur group_width
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3. Squeeze-and-Excitation (SE) - UNIQUEMENT pour RegNetY
        if self.use_se:
            self.se = SEBlock(bottleneck_channels, reduction_ratio=4) # Le rapport 4 est standard RegNetY
        
        # 4. Dernière Conv 1x1 (Projection/Expansion)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        
        # Branche principale (Main Branch)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        # *** Insertion du SE-Block ici ***
        if self.use_se:
            out = self.se(out)
            
        out = self.bn3(self.conv3(out)) # Pas de ReLU après la dernière BN
        
        # Connexion Résiduelle
        out += identity
        out = self.relu(out)
        return out

# -----------------------------------------------------------------------------
# Blocs Stem, Stage et Head (Aucun changement majeur nécessaire, juste le nettoyage)
# -----------------------------------------------------------------------------

class Stem(nn.Module):
    """
    Tige d'entrée du réseau, fixe : 3x3 conv, stride=2, output 32 canaux.
    """
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class Stage(nn.Module):
    """
    Représente une étape dans le corps du réseau (Body). 
    Séquence de d_i blocs (XBlock ou XBlock avec SE).
    """
    def __init__(self, in_channels: int, out_channels: int, depth: int, bottleneck_ratio: int, group_width: int, stride_first_block: int = 2, use_se: bool = False):
        super().__init__()
        layers = []
        
        # Block 1 (Réduction de la résolution si nécessaire/changement de canaux)
        layers.append(XBlock(
            in_channels,
            out_channels,
            bottleneck_ratio,
            group_width,
            stride=stride_first_block, # Le changement de résolution est toujours au début du stage
            use_se=use_se # Passe l'info SE
        ))
        
        # Blocs restants (d_i - 1)
        for _ in range(depth - 1):
            layers.append(XBlock(
                out_channels, 
                out_channels,
                bottleneck_ratio,
                group_width,
                stride=1, 
                use_se=use_se # Passe l'info SE
            ))
            
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)
    
class Head(nn.Module):
    """
    Tête de sortie du réseau.
    """
    def __init__(self, in_channels: int, num_classes: int = 1000):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        return x