import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import shutil # Ajout pour créer des répertoires de sauvegarde si nécessaire

class Trainer:
    def __init__(self, cfg, model: nn.Module, train_loader, val_loader, device: torch.device):
        self.cfg = cfg 
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Hyperparamètres (assurez-vous que ces clés existent dans votre YAML)
        self.lr_base = cfg.TRAIN.LR 
        self.wd = cfg.TRAIN.WEIGHT_DECAY
        self.momentum = cfg.TRAIN.MOMENTUM
        self.epochs = cfg.TRAIN.EPOCHS
        
        # Gestion du Warmup (avec valeur par défaut pour la robustesse)
        self.WARMUP_EPOCHS = cfg.TRAIN.get('WARMUP_EPOCHS', 5) # Utilisation de .get() pour une valeur par défaut
        
        # Critère et Optimiseur
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr_base,
            momentum=self.momentum,
            weight_decay=self.wd
        )
        
        # Scheduler de Learning Rate 
        self.scheduler = self._configure_scheduler()
        self.steps_per_epoch = len(train_loader)
        
        # Initialisation de la meilleure précision pour le suivi
        self.best_accuracy = 0.0

    def _configure_scheduler(self):
        # T_max est le nombre total d'époques pour la descente cosinusoïdale.
        scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.epochs - self.WARMUP_EPOCHS, # Le schedular ne commence qu'après le Warmup
            eta_min=0.0 
        )
        return scheduler

    def train_epoch(self, epoch):
        self.model.train()
        
        # Calcul du pas global et du nombre total de pas de warmup
        global_step_start = epoch * self.steps_per_epoch
        warmup_steps_total = self.WARMUP_EPOCHS * self.steps_per_epoch
        
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch+1}/{self.epochs}", dynamic_ncols=True)
        
        for i, (images, targets) in enumerate(pbar):
            global_step = global_step_start + i
            
            # 1. Gestion du Learning Rate (Warmup vs Cosine Annealing)
            current_lr = self.optimizer.param_groups[0]['lr'] # LR de base ou LR du scheduler précédent
            
            if epoch < self.WARMUP_EPOCHS:
                # Calcul progressif du LR pendant le Warmup
                new_lr = self.lr_base * (global_step / warmup_steps_total)
                current_lr = new_lr # Met à jour le LR affiché
            
            # Mise à jour du LR de l'optimiseur
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            # 2. Entraînement standard
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({'Loss': f'{running_loss / (i+1):.4f}', 'LR': f'{current_lr:.6f}'})

        # 3. Step du Cosine Annealing (uniquement après le Warmup)
        if epoch >= self.WARMUP_EPOCHS:
            # Note: Le scheduler est 'stepé' une fois par époque.
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
        for epoch in range(self.epochs):
            # 1. Entraînement
            train_loss = self.train_epoch(epoch)
            
            # 2. Validation
            val_loss, val_acc = self.validate()
            
            print(f"\n--- Époque {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ---")
            
            # 3. Sauvegarde du meilleur modèle
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                # Ne pas sauvegarder si l'entraînement est marqué comme factice (DUMMY_DATA)
                if self.cfg.TRAIN.get('SAVE_MODEL', True): 
                    self._save_checkpoint(epoch, self.best_accuracy)

    def _save_checkpoint(self, epoch, accuracy):
        """Sauvegarde les poids du modèle."""
        
        # S'assurer que le répertoire de sauvegarde existe
        save_dir = self.cfg.PATHS.MODEL_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_accuracy': accuracy,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.cfg # Sauvegarde de la configuration utilisée
        }
        
        save_path = f"{save_dir}/{self.cfg.PATHS.CHECKPOINT_NAME}"
        torch.save(checkpoint, save_path)
        print(f"Sauvegarde du meilleur modèle (Acc: {accuracy:.2f}%) vers {save_path}")