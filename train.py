"""
Training script for Ethereum Fraud Detection GNN model
"""

import os
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import create_model


class Trainer:
    """Trainer class for GNN fraud detection model."""
    
    def __init__(self, data_dir, device=None, model_type='gat', 
                 hidden_channels=128, num_heads=8, num_layers=3,
                 dropout=0.5, lr=0.001, weight_decay=5e-4,
                 batch_size=None, epochs=200):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory containing processed data
            device: PyTorch device (auto-detect if None)
            model_type: 'gat' or 'sage'
            hidden_channels: Hidden layer dimension
            num_heads: Number of attention heads (for GAT)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            batch_size: Batch size (None for full-batch)
            epochs: Maximum number of epochs
        """
        self.data_dir = data_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.optimizer = None
        self.data = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    
    def load_data(self):
        """Load processed data from files."""
        # Check if per-split features exist (new temporal splitting approach)
        train_features_path = os.path.join(self.data_dir, 'train_node_features.npy')
        val_features_path = os.path.join(self.data_dir, 'val_node_features.npy')
        test_features_path = os.path.join(self.data_dir, 'test_node_features.npy')
        
        use_per_split_features = (
            os.path.exists(train_features_path) and
            os.path.exists(val_features_path) and
            os.path.exists(test_features_path)
        )
        
        if use_per_split_features:
            print("Loading per-split features (temporal splitting)...")
            # Load per-split features
            train_features = np.load(train_features_path)
            val_features = np.load(val_features_path)
            test_features = np.load(test_features_path)
            
            # Load masks
            train_mask = np.load(os.path.join(self.data_dir, 'train_mask.npy'))
            val_mask = np.load(os.path.join(self.data_dir, 'val_mask.npy'))
            test_mask = np.load(os.path.join(self.data_dir, 'test_mask.npy'))
            
            # Create unified feature matrix using per-split features
            # Use test features as base (most complete graph)
            node_features = test_features.copy()
            # Replace with train features for train nodes
            node_features[train_mask] = train_features[train_mask]
            # Replace with val features for val nodes
            node_features[val_mask] = val_features[val_mask]
            
            # Load edge data (use test graph structure - most complete)
            edge_index = np.load(os.path.join(self.data_dir, 'edge_index.npy'))
            edge_features = np.load(os.path.join(self.data_dir, 'edge_features.npy'))
            
            # Load labels (use test labels as base)
            node_labels = np.load(os.path.join(self.data_dir, 'node_labels.npy'))
        else:
            print("Loading standard features (legacy format)...")
            # Load standard format
            node_features = np.load(os.path.join(self.data_dir, 'node_features.npy'))
            edge_index = np.load(os.path.join(self.data_dir, 'edge_index.npy'))
            edge_features = np.load(os.path.join(self.data_dir, 'edge_features.npy'))
            node_labels = np.load(os.path.join(self.data_dir, 'node_labels.npy'))
            train_mask = np.load(os.path.join(self.data_dir, 'train_mask.npy'))
            val_mask = np.load(os.path.join(self.data_dir, 'val_mask.npy'))
            test_mask = np.load(os.path.join(self.data_dir, 'test_mask.npy'))
        
        # Load metadata
        with open(os.path.join(self.data_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        y = torch.tensor(node_labels, dtype=torch.long)
        
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
        # Store per-split features if available
        if use_per_split_features:
            self.train_features = torch.tensor(train_features, dtype=torch.float32).to(self.device)
            self.val_features = torch.tensor(val_features, dtype=torch.float32).to(self.device)
            self.test_features = torch.tensor(test_features, dtype=torch.float32).to(self.device)
        else:
            self.train_features = None
            self.val_features = None
            self.test_features = None
        
        # Create PyG Data object
        self.data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        # Move to device
        self.data = self.data.to(self.device)
        
        return self.data
    
    def create_model(self):
        """Create and initialize the model."""
        in_channels = self.data.x.shape[1]
        
        model_kwargs = {
            'hidden_channels': self.hidden_channels,
            'out_channels': 2,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'use_batch_norm': True
        }
        
        if self.model_type.lower() in ['gat', 'tgn', 'tgat']:
            model_kwargs['num_heads'] = self.num_heads
        
        self.model = create_model(
            model_type=self.model_type,
            in_channels=in_channels,
            **model_kwargs
        )
        
        self.model = self.model.to(self.device)
        
        return self.model
    
    def create_optimizer(self):
        """Create optimizer and loss function."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Use weighted loss for imbalanced classes
        # Use fixed 10:1 ratio (fraud:non-fraud) to reduce false positives
        train_labels = self.data.y[self.data.train_mask]
        num_pos = train_labels.sum().item()
        num_neg = (train_labels == 0).sum().item()
        
        if num_pos > 0 and num_neg > 0:
            # Use fixed 10:1 ratio instead of inverse frequency
            pos_weight = torch.tensor([10.0], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        return self.optimizer
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Use train features if available (temporal splitting)
        if self.train_features is not None:
            x_input = self.train_features
        else:
            x_input = self.data.x
        
        # Forward pass (pass edge attributes for temporal models)
        out = self.model(x_input, self.data.edge_index, self.data.edge_attr)
        
        # Compute loss on training nodes
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            # Binary classification
            loss = self.criterion(out[self.data.train_mask, 1], 
                                 self.data.y[self.data.train_mask].float())
        else:
            # Multi-class classification
            loss = self.criterion(out[self.data.train_mask], 
                                 self.data.y[self.data.train_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def recall_at_k(self, y_true, y_scores, k_values=[100, 500, 1000]):
        """
        Calculate Recall@K for different K values.
        
        Args:
            y_true: True binary labels
            y_scores: Predicted scores (higher = more likely to be fraud)
            k_values: List of K values to compute recall at
        
        Returns:
            Dictionary of recall@k for each k
        """
        # Sort by scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        total_positives = y_true.sum()
        if total_positives == 0:
            return {f'recall@{k}': 0.0 for k in k_values}
        
        recall_at_k = {}
        for k in k_values:
            if k > len(y_true):
                k = len(y_true)
            # Number of true positives in top K
            tp_at_k = sorted_labels[:k].sum()
            recall_at_k[f'recall@{k}'] = tp_at_k / total_positives
        
        return recall_at_k
    
    def evaluate(self, mask, return_scores=False):
        """
        Evaluate model on given mask using ranking metrics.
        
        Args:
            mask: Node mask for evaluation
            return_scores: Whether to return raw scores and labels
        
        Returns:
            Dictionary with ranking metrics: loss, roc_auc, pr_auc, recall@k
        """
        self.model.eval()
        
        with torch.no_grad():
            # Use appropriate features based on mask (temporal splitting)
            if self.train_features is not None:
                # Determine which features to use based on mask
                # Check if mask matches train/val/test masks
                mask_np = mask.cpu().numpy()
                train_mask_np = self.data.train_mask.cpu().numpy()
                val_mask_np = self.data.val_mask.cpu().numpy()
                test_mask_np = self.data.test_mask.cpu().numpy()
                
                if np.array_equal(mask_np, train_mask_np):
                    x_input = self.train_features
                elif np.array_equal(mask_np, val_mask_np):
                    x_input = self.val_features
                elif np.array_equal(mask_np, test_mask_np):
                    x_input = self.test_features
                else:
                    # Mixed mask, use test features as default
                    x_input = self.test_features
            else:
                x_input = self.data.x
            
            # Pass edge attributes for temporal models
            out = self.model(x_input, self.data.edge_index, self.data.edge_attr)
            
            # Get raw scores (logits for fraud class)
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                # For BCE, we use the logit for class 1 (fraud)
                y_scores = out[:, 1].cpu().numpy()
            else:
                # For CrossEntropy, convert to probabilities then use fraud class
                probs = F.softmax(out, dim=1)
                y_scores = probs[:, 1].cpu().numpy()
            
            labels = self.data.y.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            if mask_np.sum() == 0:
                return {}
            
            y_true = labels[mask_np]
            y_scores_masked = y_scores[mask_np]
            
            # Compute metrics
            metrics = {}
            
            # ROC-AUC
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores_masked)
            except:
                metrics['roc_auc'] = None
            
            # PR-AUC (Precision-Recall AUC)
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_scores_masked)
                metrics['pr_auc'] = auc(recall, precision)
            except:
                metrics['pr_auc'] = None
            
            # Recall@K
            recall_at_k = self.recall_at_k(y_true, y_scores_masked, k_values=[100, 500, 1000])
            metrics.update(recall_at_k)
            
            # Loss
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                loss = self.criterion(out[mask, 1], self.data.y[mask].float()).item()
            else:
                loss = self.criterion(out[mask], self.data.y[mask]).item()
            
            metrics['loss'] = loss
            
            if return_scores:
                return metrics, y_scores_masked, y_true
            return metrics
    
    def train(self, save_dir='./checkpoints'):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training"):
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            train_metrics = self.evaluate(self.data.train_mask)
            val_metrics = self.evaluate(self.data.val_mask)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_auc'].append(train_metrics.get('roc_auc', 0))
            self.history['val_auc'].append(val_metrics.get('roc_auc', 0))
            
            # Print progress
            if epoch % 10 == 0 or epoch == 1:
                train_auc = train_metrics.get('roc_auc', 0) or 0
                train_pr_auc = train_metrics.get('pr_auc', 0) or 0
                train_recall_100 = train_metrics.get('recall@100', 0) or 0
                val_auc = val_metrics.get('roc_auc', 0) or 0
                val_pr_auc = val_metrics.get('pr_auc', 0) or 0
                val_recall_100 = val_metrics.get('recall@100', 0) or 0
                print(f"Epoch {epoch:3d}/{self.epochs} - Train: Loss={train_loss:.4f}, ROC-AUC={train_auc:.4f}, PR-AUC={train_pr_auc:.4f}, R@100={train_recall_100:.4f} | Val: Loss={val_metrics['loss']:.4f}, ROC-AUC={val_auc:.4f}, PR-AUC={val_pr_auc:.4f}, R@100={val_recall_100:.4f}")
        
        # Save model at last epoch
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_roc_auc': val_metrics.get('roc_auc', 0),
            'model_config': {
                'model_type': self.model_type,
                'in_channels': self.data.x.shape[1],
                'hidden_channels': self.hidden_channels,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, os.path.join(save_dir, 'best_model.pt'))
        
        # Final evaluation on test set
        test_metrics, y_scores, y_true = self.evaluate(
            self.data.test_mask, return_scores=True
        )
        
        # Print ranking metrics
        roc_auc = test_metrics.get('roc_auc', 0) or 0
        pr_auc = test_metrics.get('pr_auc', 0) or 0
        recall_100 = test_metrics.get('recall@100', 0) or 0
        recall_500 = test_metrics.get('recall@500', 0) or 0
        recall_1000 = test_metrics.get('recall@1000', 0) or 0
        print(f"\nTest - Loss: {test_metrics['loss']:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"Recall@100: {recall_100:.4f}, Recall@500: {recall_500:.4f}, Recall@1000: {recall_1000:.4f}")
        
        # Save history
        with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.history, f)
        
        # Save evaluation summary
        eval_summary = {
            'test_metrics': test_metrics
        }
        with open(os.path.join(save_dir, 'evaluation_summary.pkl'), 'wb') as f:
            pickle.dump(eval_summary, f)
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        return test_metrics
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # AUC curve
        axes[1].plot(self.history['train_auc'], label='Train AUC')
        axes[1].plot(self.history['val_auc'], label='Val AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Training and Validation AUC')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Train GNN fraud detection model')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                        help='Directory containing processed data')
    parser.add_argument('--model', type=str, default='tgn', choices=['tgn', 'tgat', 'gat', 'sage'],
                        help='Model type: tgn (default, temporal GAT), tgat (temporal GAT), gat, or sage')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden channels')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads (for GAT)')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu), auto-detect if None')
    
    args = parser.parse_args()
    
    # Create device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create trainer
    trainer = Trainer(
        data_dir=args.data_dir,
        device=device,
        model_type=args.model,
        hidden_channels=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs
    )
    
    # Load data
    trainer.load_data()
    
    # Create model
    trainer.create_model()
    
    # Create optimizer
    trainer.create_optimizer()
    
    # Train
    trainer.train(save_dir=args.save_dir)


if __name__ == '__main__':
    main()

