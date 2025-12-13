import torch.nn as nn
import torch 
import math
class Stem(nn.Module):
    """
    Tige d'entrée du réseau, fixe : 3x3 conv, stride=2, output 32 canaux.
    """
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        # Conv, BatchNorm et ReLU sont appliqués séquentiellement
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class XBlock(nn.Module):
    """
    Bloc résiduel Bottleneck avec Group Convolution (X Block).
    Basé sur Figure 4a.
    """
    expansion = 1 # Pas d'expansion de la largeur d'entrée/sortie du bloc
    
    def __init__(self, in_channels: int, out_channels: int, bottleneck_ratio: int, group_width: int, stride: int = 1):
        super().__init__()
        
        # Le nombre de canaux à l'intérieur du bottleneck est out_channels / bottleneck_ratio
        # Le rapport de g / b est la taille du groupe par canal (channels per group)
        bottleneck_channels = out_channels // bottleneck_ratio
        num_groups = bottleneck_channels // group_width
        
        # Vérification de la compatibilité (requis par l'article, doit être géré en amont)
        if num_groups == 0:
             # C'est une erreur de configuration (w/b doit être divisible par g)
             num_groups = 1 # Fail-safe minimal
        elif bottleneck_channels % group_width != 0:
            # Pour la fidélité, il faudrait ajuster group_width ou bottleneck_channels
            # Nous assumons que les paramètres RegNet sont divisibles par 8 et que g=8 ou 16 est utilisé.
            # Ici, nous nous assurons que num_groups est au moins 1.
            num_groups = max(1, math.floor(bottleneck_channels / group_width))
        # Shortcut (Branche latérale résiduelle)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Si la résolution (stride!=1) ou le nombre de canaux change, on utilise 
            # une conv 1x1 avec le même stride que le bloc principal.
            # (Basé sur Figure 4b)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 1. Première Conv 1x1 (Réduction de la largeur)
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
        
        # 3. Dernière Conv 1x1 (Projection/Expansion de la largeur)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Enregistre l'entrée pour la connexion résiduelle
        identity = self.shortcut(x)
        
        # Branche principale (Main Branch)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) # Pas de ReLU après la dernière BN
        # Connexion Résiduelle
        out += identity
        out = self.relu(out)
        return out
    
class Stage(nn.Module):
    """
    Représente une étape dans le corps du réseau (Body). 
    C'est une séquence de d_i blocs.
    """
    def __init__(self, in_channels: int, out_channels: int, depth: int, bottleneck_ratio: int, group_width: int, stride_first_block: int = 2):
        super().__init__()
        layers = []
        
        # Le premier bloc a stride=2 pour réduire la résolution (sauf pour le Stage 1 après le Stem)
        # La résolution est déjà réduite par le Stem (stride=2), donc le Stage 1 peut commencer par stride=1.
        # En réalité, la réduction de résolution a lieu entre les stages.
        
        # Le premier bloc du stage i>1 utilise stride=2, comme illustré dans Figure 3c.
        # Ici, nous mettons stride=2 pour la première couche s'il y a un changement de canaux/résolution.
        
        # Block 1 (Réduction de la résolution si nécessaire/changement de canaux)
        layers.append(XBlock(
            in_channels,
            out_channels,
            bottleneck_ratio,
            group_width,
            stride=stride_first_block # Le changement de résolution est toujours au début du stage
        ))
        
        # Blocs restants (d_i - 1)
        for _ in range(depth - 1):
            layers.append(XBlock(
                out_channels, # L'entrée est la sortie du bloc précédent
                out_channels,
                bottleneck_ratio,
                group_width,
                stride=1 # Pas de réduction de résolution
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
        # Global Average Pooling pour convertir la carte de caractéristiques en vecteur
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully Connected Layer pour la classification (ImageNet a 1000 classes)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # Flatten (Aplatir) la sortie (batch, channels, 1, 1) en (batch, channels)
        x = self.fc(x)
        return x