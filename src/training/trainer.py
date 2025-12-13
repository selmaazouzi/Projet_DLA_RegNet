import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
# Importer les outils de configuration (supposons un objet 'cfg' chargé du YAML)

class Trainer:
    def __init__(self, cfg, model: nn.Module, train_loader, val_loader, device: torch.device):
        # Configuration (du fichier YAML)
        self.cfg = cfg 
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device= device
        
        # Hyperparamètres
        self.lr_base = cfg.TRAIN.LR  # e.g., 0.1
        self.wd = cfg.TRAIN.WEIGHT_DECAY # e.g., 5e-5
        self.momentum = cfg.TRAIN.MOMENTUM # e.g., 0.9
        self.epochs = cfg.TRAIN.EPOCHS # e.g., 100
        
        # Critère et Optimiseur
        # --- CORRECTION CRITIQUE : Utiliser .to(self.device) au lieu de .cuda() ---
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr_base,
            momentum=self.momentum,
            weight_decay=self.wd
        )
        
        # Scheduler de Learning Rate (Logique Cruciale)
        self.scheduler = self._configure_scheduler()
        
        # Nombre de batchs/itération par époque
        self.steps_per_epoch = len(train_loader)

    def _configure_scheduler(self):
        # Selon l'article, on utilise un schedule en demi-période cosinusoïdale.
        T_max_epochs = self.epochs
        
        scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=T_max_epochs, 
            eta_min=0.0  # Taux d'apprentissage minimal
        )
        
        return scheduler

    def train_epoch(self, epoch):
        self.model.train()
        
        # Ajout de la barre de progression
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch+1}/{self.epochs}", dynamic_ncols=True)
        
        WARMUP_EPOCHS = self.cfg.TRAIN.WARMUP_EPOCHS if 'WARMUP_EPOCHS' in self.cfg.TRAIN else 5
        warmup_steps = WARMUP_EPOCHS * self.steps_per_epoch
        
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(pbar):
            global_step = epoch * self.steps_per_epoch + i
            
            # 1. Ajustement du Learning Rate (Warmup et Cosine)
            if epoch < WARMUP_EPOCHS:
                new_lr = self.lr_base * (global_step / warmup_steps)
            else:
                new_lr = self.optimizer.param_groups[0]['lr']
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # 2. Entraînement standard
            # Les données sont déplacées vers le bon dispositif (self.device)
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({'Loss': f'{running_loss / (i+1):.4f}', 'LR': f'{new_lr:.6f}'})

        # 3. Step du Cosine Annealing (si le Warmup est terminé)
        if epoch >= WARMUP_EPOCHS:
            self.scheduler.step()
            
        return running_loss / self.steps_per_epoch

    def validate(self):
        """
        Évalue le modèle sur l'ensemble de validation et retourne la perte et la précision Top-1.
        """
        self.model.eval() 
        
        total_loss = 0
        correct_top1 = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validate", dynamic_ncols=True):
                # Les données sont déplacées vers le bon dispositif (self.device)
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * images.size(0)
                
                # Calcul de la précision Top-1
                _, predicted = outputs.max(1)
                correct_top1 += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        top1_accuracy = correct_top1 / total_samples * 100.0
        
        return avg_loss, top1_accuracy

    def train(self):
        """
        Gère la boucle complète d'entraînement et de validation sur toutes les époques.
        """
        best_accuracy = 0.0
        
        for epoch in range(self.epochs):
            # 1. Entraînement
            train_loss = self.train_epoch(epoch)
            
            # 2. Validation
            val_loss, val_acc = self.validate()
            
            print(f"\n--- Époque {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ---")
            
            # 3. Sauvegarde du meilleur modèle
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self._save_checkpoint(epoch, best_accuracy)

    def _save_checkpoint(self, epoch, accuracy):
        """Sauvegarde les poids du modèle."""
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_accuracy': accuracy,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        
        save_path = f"{self.cfg.PATHS.MODEL_SAVE_DIR}/{self.cfg.PATHS.CHECKPOINT_NAME}"
        torch.save(checkpoint, save_path)
        print(f"Sauvegarde du meilleur modèle (Acc: {accuracy:.2f}%) vers {save_path}")