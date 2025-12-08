import torch
import torch.nn as nn
import numpy as np

# -----------------------------------------------------------------------------
# 1. LE GÉNÉRATEUR DE PARAMÈTRES (Cœur de l'innovation du papier)
# Référence : Section 3.3 "The RegNet Design Space", Équations 2 à 4
# -----------------------------------------------------------------------------

def generate_regnet_parameters(initial_width, slope, quantized_param, depth, group_width_divisor=8):
    """
    Génère les largeurs et profondeurs par stage.
    Args:
        initial_width (w0): Largeur initiale [cite: 317]
        slope (wa): Pente de la ligne linéaire [cite: 317]
        quantized_param (wm): Paramètre de quantification exponentielle [cite: 361]
        depth (d): Profondeur totale du réseau
    """
    # Eq (2): Progression linéaire théorique u_j = w0 + wa * j
    indices = np.arange(depth)
    u_j = initial_width + slope * indices

    # Eq (3): Calcul des "steps" logarithmiques s_j
    # On calcule s_j tel que u_j = w0 * wm^s_j
    s_j = np.log(u_j / initial_width) / np.log(quantized_param)
    s_j = np.round(s_j)

    # Eq (4): Largeurs quantifiées finales w_j
    w_j = initial_width * np.power(quantized_param, s_j)

    # Règle technique : Arrondir pour être divisible par 8 (pour efficacité GPU)
    w_j = (np.round(w_j / group_width_divisor) * group_width_divisor).astype(int)

    # On transforme la liste des blocs en (largeur, nombre_blocs) par stage
    # numpy.unique nous donne les valeurs uniques et leur fréquence (count = depth du stage)
    unique_widths, counts = np.unique(w_j, return_counts=True)
    
    # Tri pour garantir l'ordre croissant des largeurs
    sorted_indices = np.argsort(unique_widths)
    unique_widths = unique_widths[sorted_indices]
    counts = counts[sorted_indices]

    return unique_widths.tolist(), counts.tolist()


# -----------------------------------------------------------------------------
# 2. LES COMPOSANTS DU RÉSEAU (Blocs)
# Référence : Figure 3 (Structure globale) et Figure 4 (X Block)
# -----------------------------------------------------------------------------

class Stem(nn.Module):
    """
    La tige (Stem) : Conv 3x3 simple, stride 2, 32 channels.
    Référence : Figure 3a et 3b [cite: 179]
    """
    def __init__(self, in_channels=3, out_channels=32):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class XBlock(nn.Module):
    """
    Le bloc standard X basé sur un Bottleneck avec Group Convolution.
    Référence : Figure 4 [cite: 203-206]
    Structure : 1x1 conv -> 3x3 group conv -> 1x1 conv
    """
    def __init__(self, in_w, out_w, stride, group_width):
        super(XBlock, self).__init__()
        
        # Le papier recommande bottleneck_ratio b=1 (donc pas de réduction interne) [cite: 468]
        inter_w = out_w 
        
        # Calcul du nombre de groupes (g)
        groups = inter_w // group_width
        
        # Règle de compatibilité (Appendix D) : on ajuste groups pour qu'il divise inter_w [cite: 1125-1126]
        if groups == 0: groups = 1
        while inter_w % groups != 0:
            groups -= 1
        
        # 1x1 Conv (Projection)
        self.conv1 = nn.Conv2d(in_w, inter_w, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_w)
        
        # 3x3 Group Conv (C'est ici que se joue l'efficacité de RegNet)
        self.conv2 = nn.Conv2d(inter_w, inter_w, kernel_size=3, stride=stride, 
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_w)
        
        # 1x1 Conv (Expansion)
        self.conv3 = nn.Conv2d(inter_w, out_w, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_w)
        
        self.relu = nn.ReLU(inplace=True)

        # Connection Résiduelle (Shortcut)
        # Si les dimensions changent (stride > 1 ou changement de largeur), on adapte x
        self.shortcut = nn.Sequential()
        if stride != 1 or in_w != out_w:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_w, out_w, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_w)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


# -----------------------------------------------------------------------------
# 3. LE MODÈLE COMPLET (RegNetX)
# -----------------------------------------------------------------------------

class RegNetX(nn.Module):
    """
    Assembleur final.
    """
    def __init__(self, initial_width, slope, quantized_param, depth, group_width, num_classes=10):
        super(RegNetX, self).__init__()
        
        # 1. On génère la structure via les maths
        self.widths, self.depths = generate_regnet_parameters(initial_width, slope, quantized_param, depth)
        
        # 2. Le Stem [cite: 179]
        self.in_planes = 32
        self.stem = Stem(in_channels=3, out_channels=32)
        
        # 3. Le Body (Sequence de 4 stages) [cite: 180, 190]
        self.stages = nn.ModuleList()
        for i, (stage_width, stage_depth) in enumerate(zip(self.widths, self.depths)):
            layers = []
            for j in range(stage_depth):
                # Le premier bloc du stage gère le stride (réduction de résolution) [cite: 181]
                # Note: Sur le premier stage (i=0), on reste souvent en stride 1 ou 2 selon le dataset.
                stride = 2 if j == 0 else 1
                
                # Petit ajustement pour CIFAR-10 (petites images) : Stride 1 au stage 0
                if i == 0: stride = 1 
                
                layers.append(XBlock(self.in_planes, stage_width, stride, group_width))
                self.in_planes = stage_width
            
            self.stages.append(nn.Sequential(*layers))

        # 4. Le Head [cite: 179]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_planes, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# -----------------------------------------------------------------------------
# TEST RAPIDE (Pour vérifier que ça marche)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Paramètres proches de RegNetX-200MF
    # w0=24, wa=36, wm=2.5, d=13 [cite: 498]
    model = RegNetX(initial_width=24, slope=36, quantized_param=2.5, depth=13, group_width=8)
    
    print("Structure générée (Largeurs par stage):", model.widths)
    print("Structure générée (Profondeur par stage):", model.depths)
    
    # Test avec une image factice
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print("Taille sortie:", output.shape) # Doit être [1, 10]
