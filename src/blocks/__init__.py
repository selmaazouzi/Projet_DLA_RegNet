# Fichier : src/blocks/__init__.py

# L'existence de ce fichier seul suffit à faire de 'blocks' un package.

# Pour faciliter l'importation, on importe les classes directement ici.
# Cela permet d'écrire: 'from src.blocks import XBlock' au lieu de 'from src.blocks.xblock import XBlock'

from .xblock import Stem, XBlock, Stage, Head