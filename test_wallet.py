"""
Wallet Fraud Detection using Web3
This script fetches transactions for a given wallet address using Web3
and uses the trained GNN model to predict fraud probability.
"""

import os
import argparse
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Web3 imports
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    print("Warning: web3 not installed. Install with: pip install web3")

from model import create_model


class WalletAnalyzer:
    """
    Analyzer for wallet addresses using Web3 and trained GNN model.
    """
    
    def __init__(self, model_path, data_dir, rpc_url=None, device=None):
        """
        Initialize wallet analyzer.
        
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Directory containing preprocessed data (for scalers)
            rpc_url: Ethereum RPC URL (e.g., Infura, Alchemy)
            device: PyTorch device
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Web3 if available
        self.web3 = None
        if HAS_WEB3:
            if rpc_url:
                self.web3 = Web3(Web3.HTTPProvider(rpc_url))
                # Add PoA middleware for networks like BSC, Polygon
                try:
                    self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
                except:
                    pass
                
                if self.web3.is_connected():
                    print(f"Connected to Ethereum RPC: {rpc_url}")
                else:
                    print("Warning: Could not connect to Ethereum RPC")
        
        # Load scalers and metadata
        self.load_scalers()
        
        # Load model
        self.load_model()
        
        print(f"Wallet analyzer initialized on device: {self.device}")
    
    def load_scalers(self):
        """Load feature scalers from preprocessing."""
        print("Loading scalers...")
        
        with open(os.path.join(self.data_dir, 'scaler.pkl'), 'rb') as f:
            self.node_scaler = pickle.load(f)
        
        with open(os.path.join(self.data_dir, 'edge_scaler.pkl'), 'rb') as f:
            self.edge_scaler = pickle.load(f)
        
        with open(os.path.join(self.data_dir, 'feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"Loaded scalers for {len(self.feature_names)} features")
    
    def load_model(self):
        """Load trained model."""
        print(f"Loading model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint['model_config']
        
        self.model = create_model(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Try to load optimal threshold
        checkpoint_dir = os.path.dirname(self.model_path)
        threshold_file = os.path.join(checkpoint_dir, 'optimal_threshold.txt')
        self.threshold = 0.5  # default
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, 'r') as f:
                    self.threshold = float(f.read().strip())
                print(f"Loaded optimal threshold: {self.threshold:.4f}")
            except:
                print(f"Using default threshold: 0.5")
        else:
            print(f"Optimal threshold not found, using default: 0.5")
        
        print(f"Model loaded: {model_config['model_type'].upper()}")
        print(f"  Validation AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
        print(f"  Classification threshold: {self.threshold:.4f}")
    
    def fetch_wallet_transactions(self, wallet_address, max_transactions=1000):
        """
        Fetch transactions for a wallet address using Web3.
        
        Args:
            wallet_address: Ethereum wallet address
            max_transactions: Maximum number of transactions to fetch
        
        Returns:
            List of transaction dictionaries
        """
        if not HAS_WEB3 or not self.web3 or not self.web3.is_connected():
            raise RuntimeError(
                "Web3 not available or not connected. "
                "Please provide RPC_URL or install web3: pip install web3"
            )
        
        wallet_address = Web3.to_checksum_address(wallet_address)
        print(f"\nFetching transactions for wallet: {wallet_address}")
        print(f"Maximum transactions: {max_transactions}")
        
        transactions = []
        seen_hashes = set()
        
        # Get current block number
        try:
            current_block = self.web3.eth.block_number
            print(f"Current block: {current_block}")
        except Exception as e:
            print(f"Warning: Could not get current block: {e}")
            return transactions
        
        # Fetch transactions in batches
        batch_size = 1000
        start_block = max(0, current_block - 100000)  # Last ~100k blocks
        
        print(f"Scanning blocks from {start_block} to {current_block}...")
        
        try:
            # Method 1: Scan recent blocks (may be slow)
            # For production, use a blockchain explorer API or indexer
            for block_num in range(start_block, current_block + 1, batch_size):
                if len(transactions) >= max_transactions:
                    break
                
                try:
                    block = self.web3.eth.get_block(min(block_num + batch_size - 1, current_block), full_transactions=True)
                    
                    for tx in block.transactions:
                        if len(transactions) >= max_transactions:
                            break
                        
                        tx_hash = tx.hash.hex()
                        if tx_hash in seen_hashes:
                            continue
                        
                        # Check if wallet is sender or receiver
                        from_addr = tx['from']
                        to_addr = tx.get('to')
                        
                        if from_addr and from_addr.lower() == wallet_address.lower():
                            transactions.append({
                                'TxHash': tx_hash,
                                'From': from_addr,
                                'To': to_addr,
                                'Value': self.web3.from_wei(tx['value'], 'ether'),
                                'BlockHeight': block_num,
                                'TimeStamp': block['timestamp'],
                                'isError': 0  # Will need to check receipt
                            })
                            seen_hashes.add(tx_hash)
                        
                        elif to_addr and to_addr.lower() == wallet_address.lower():
                            transactions.append({
                                'TxHash': tx_hash,
                                'From': from_addr,
                                'To': to_addr,
                                'Value': self.web3.from_wei(tx['value'], 'ether'),
                                'BlockHeight': block_num,
                                'TimeStamp': block['timestamp'],
                                'isError': 0
                            })
                            seen_hashes.add(tx_hash)
                except Exception as e:
                    # Skip blocks that can't be fetched
                    continue
        except Exception as e:
            print(f"Warning: Error fetching transactions: {e}")
        
        print(f"Found {len(transactions)} transactions")
        return transactions[:max_transactions]
    
    def fetch_wallet_transactions_from_api(self, wallet_address, api_key=None, max_transactions=1000):
        """
        Alternative method: Fetch transactions using blockchain explorer API.
        More efficient for large-scale analysis.
        
        Example APIs: Etherscan, Alchemy, Infura
        """
        print(f"\nFetching transactions via API for wallet: {wallet_address}")
        print("Note: This is a placeholder. Implement with your preferred API.")
        
        # Example with Etherscan-like API
        # transactions = []
        # response = requests.get(f"https://api.etherscan.io/api?module=account&action=txlist&address={wallet_address}&apikey={api_key}")
        # ... parse response ...
        
        return []
    
    def compute_node_features_single(self, wallet_address, transactions):
        """
        Compute node features for a single wallet from its transactions.
        Uses the same feature engineering as preprocessing.
        """
        if len(transactions) == 0:
            # Return zero features if no transactions
            n_features = len(self.feature_names)
            return np.zeros(n_features, dtype=np.float32)
        
        # Initialize feature arrays
        features = defaultdict(float)
        
        in_count = 0
        out_count = 0
        total_in = 0.0
        total_out = 0.0
        max_in = 0.0
        max_out = 0.0
        error_count = 0
        
        first_ts = 1e12
        last_ts = 0
        unique_in_neighbors = set()
        unique_out_neighbors = set()
        
        # Process transactions
        for tx in transactions:
            from_addr = tx['From']
            to_addr = tx.get('To')
            value = abs(float(tx.get('Value', 0)))
            timestamp = int(tx.get('TimeStamp', 0))
            is_error = int(tx.get('isError', 0))
            
            is_sender = from_addr and from_addr.lower() == wallet_address.lower()
            is_receiver = to_addr and to_addr.lower() == wallet_address.lower()
            
            if is_sender:
                out_count += 1
                total_out += value
                if value > max_out:
                    max_out = value
                if to_addr:
                    unique_out_neighbors.add(to_addr)
            
            if is_receiver:
                in_count += 1
                total_in += value
                if value > max_in:
                    max_in = value
                if from_addr:
                    unique_in_neighbors.add(from_addr)
            
            if is_error:
                error_count += 1
            
            if timestamp > 0:
                if timestamp < first_ts:
                    first_ts = timestamp
                if timestamp > last_ts:
                    last_ts = timestamp
        
        # Compute derived features (matching preprocessing logic)
        total_degree = in_count + out_count
        
        # Basic counts
        features['in_degree'] = float(in_count)
        features['out_degree'] = float(out_count)
        
        # Values
        features['log_total_in_value'] = np.log1p(total_in)
        features['log_total_out_value'] = np.log1p(total_out)
        
        avg_in = total_in / in_count if in_count > 0 else 0.0
        avg_out = total_out / out_count if out_count > 0 else 0.0
        features['log_avg_in_value'] = np.log1p(avg_in)
        features['log_avg_out_value'] = np.log1p(avg_out)
        features['log_max_in_value'] = np.log1p(max_in)
        features['log_max_out_value'] = np.log1p(max_out)
        
        # Neighbors
        features['unique_in_neighbors'] = float(len(unique_in_neighbors))
        features['unique_out_neighbors'] = float(len(unique_out_neighbors))
        
        # Temporal
        if first_ts < 1e11:
            span_seconds = max(0, last_ts - first_ts)
            span_days = span_seconds / 86400.0
            features['transaction_span_days'] = span_days
        else:
            features['transaction_span_days'] = 0.0
        
        if features['transaction_span_days'] > 0:
            features['tx_frequency'] = total_degree / features['transaction_span_days']
        else:
            features['tx_frequency'] = float(total_degree)
        
        # Ratios
        features['degree_ratio'] = out_count / (total_degree + 1e-8)
        features['value_ratio'] = total_out / (total_in + total_out + 1e-8)
        
        # Error rate
        features['error_rate'] = error_count / (total_degree + 1e-8)
        
        # Structural features (will be zeros for single node - need subgraph)
        features['log_pagerank'] = 0.0  # Would need subgraph
        features['clustering'] = 0.0
        features['in_degree_centrality'] = 0.0
        features['out_degree_centrality'] = 0.0
        
        # Build feature vector in correct order
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(features.get(name, 0.0))
        
        return np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    
    def build_subgraph(self, wallet_address, transactions, max_neighbors=100):
        """
        Build a subgraph around the wallet address including neighbors.
        """
        if len(transactions) == 0:
            return None
        
        # Collect all unique addresses in transactions
        all_addresses = set()
        all_addresses.add(wallet_address.lower())
        
        for tx in transactions:
            if tx.get('From'):
                all_addresses.add(tx['From'].lower())
            if tx.get('To'):
                all_addresses.add(tx['To'].lower())
        
        # Limit to max_neighbors + 1 (wallet + neighbors)
        if len(all_addresses) > max_neighbors + 1:
            # Keep wallet and most connected neighbors
            neighbor_counts = defaultdict(int)
            for tx in transactions:
                from_addr = tx.get('From', '').lower()
                to_addr = tx.get('To', '').lower()
                if from_addr != wallet_address.lower():
                    neighbor_counts[from_addr] += 1
                if to_addr and to_addr != wallet_address.lower():
                    neighbor_counts[to_addr] += 1
            
            # Sort by connection count
            sorted_neighbors = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)
            top_neighbors = set([wallet_address.lower()] + [n[0] for n in sorted_neighbors[:max_neighbors]])
            all_addresses = all_addresses.intersection(top_neighbors)
        
        # Create mapping
        addr_list = list(all_addresses)
        addr2idx = {addr: idx for idx, addr in enumerate(addr_list)}
        
        # Build edge index
        edge_list = []
        for tx in transactions:
            from_addr = tx.get('From', '').lower()
            to_addr = tx.get('To', '').lower()
            
            if from_addr in addr2idx and to_addr in addr2idx:
                edge_list.append([addr2idx[from_addr], addr2idx[to_addr]])
        
        if len(edge_list) == 0:
            return None
        
        edge_index = np.array(edge_list).T
        
        return {
            'addresses': addr_list,
            'addr2idx': addr2idx,
            'edge_index': edge_index,
            'target_idx': addr2idx[wallet_address.lower()]
        }
    
    def predict_wallet(self, wallet_address, transactions=None, use_subgraph=False):
        """
        Predict fraud probability for a wallet address.
        
        Args:
            wallet_address: Wallet address to analyze
            transactions: List of transactions (if None, will fetch via Web3)
            use_subgraph: Whether to use subgraph approach (more accurate but slower)
        
        Returns:
            Dictionary with prediction results
        """
        # Fetch transactions if not provided
        if transactions is None:
            if not HAS_WEB3 or not self.web3:
                raise RuntimeError("Web3 not available. Please provide transactions or set up RPC_URL")
            transactions = self.fetch_wallet_transactions(wallet_address)
        
        if len(transactions) == 0:
            return {
                'wallet_address': wallet_address,
                'fraud_probability': 0.5,
                'prediction': 'Unknown',
                'num_transactions': 0,
                'confidence': 'Low',
                'message': 'No transactions found'
            }
        
        wallet_address = wallet_address.lower()
        
        if use_subgraph:
            # Build subgraph
            subgraph = self.build_subgraph(wallet_address, transactions)
            if subgraph is None:
                use_subgraph = False
        
        if use_subgraph:
            # Predict using subgraph
            # This is more complex - would need to compute features for all nodes
            # For now, fall back to single node approach
            pass
        
        # Single node approach: compute features for wallet only
        node_features = self.compute_node_features_single(wallet_address, transactions)
        
        # Scale features
        node_features_scaled = self.node_scaler.transform(node_features)
        
        # Create minimal graph (single node)
        x = torch.tensor(node_features_scaled, dtype=torch.float32).to(self.device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            out = self.model(x, edge_index)
            
            if out.shape[1] == 2:
                probs = F.softmax(out, dim=1)[0].cpu().numpy()
                fraud_prob = float(probs[1])
            else:
                fraud_prob = float(torch.sigmoid(out[0]).cpu().numpy())
        
        # Determine prediction and confidence
        is_fraud = fraud_prob > self.threshold
        confidence = 'High' if abs(fraud_prob - self.threshold) > 0.3 else 'Medium' if abs(fraud_prob - self.threshold) > 0.1 else 'Low'
        
        result = {
            'wallet_address': wallet_address,
            'fraud_probability': round(fraud_prob, 4),
            'normal_probability': round(1 - fraud_prob, 4),
            'prediction': 'Fraudulent' if is_fraud else 'Normal',
            'confidence': confidence,
            'num_transactions': len(transactions),
            'num_incoming': sum(1 for tx in transactions if tx.get('To', '').lower() == wallet_address),
            'num_outgoing': sum(1 for tx in transactions if tx.get('From', '').lower() == wallet_address),
            'total_value_received': sum(abs(float(tx.get('Value', 0))) for tx in transactions if tx.get('To', '').lower() == wallet_address),
            'total_value_sent': sum(abs(float(tx.get('Value', 0))) for tx in transactions if tx.get('From', '').lower() == wallet_address),
        }
        
        return result
    
    def analyze_and_print(self, wallet_address, transactions=None, use_subgraph=False):
        """Analyze wallet and print results."""
        result = self.predict_wallet(wallet_address, transactions, use_subgraph)
        
        print("\n" + "=" * 60)
        print("Wallet Fraud Analysis Results")
        print("=" * 60)
        print(f"Wallet Address: {result['wallet_address']}")
        print(f"\nPrediction: {result['prediction']}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.2f}%)")
        print(f"Normal Probability: {result['normal_probability']:.4f} ({result['normal_probability']*100:.2f}%)")
        print(f"Confidence: {result['confidence']}")
        print(f"\nTransaction Statistics:")
        print(f"  Total Transactions: {result['num_transactions']}")
        print(f"  Incoming: {result['num_incoming']}")
        print(f"  Outgoing: {result['num_outgoing']}")
        print(f"  Total Received: {result['total_value_received']:.6f} ETH")
        print(f"  Total Sent: {result['total_value_sent']:.6f} ETH")
        print("=" * 60)
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Analyze wallet for fraud using trained model')
    parser.add_argument('wallet_address', type=str, help='Ethereum wallet address to analyze')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                        help='Directory containing processed data and scalers')
    parser.add_argument('--rpc_url', type=str, default=None,
                        help='Ethereum RPC URL (e.g., https://mainnet.infura.io/v3/YOUR_KEY)')
    parser.add_argument('--transactions_file', type=str, default=None,
                        help='Path to CSV file with transactions (optional, skips Web3 fetch)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu), auto-detect if None')
    
    args = parser.parse_args()
    
    # Load transactions from file if provided
    transactions = None
    if args.transactions_file:
        import pandas as pd
        df = pd.read_csv(args.transactions_file)
        transactions = df.to_dict('records')
        print(f"Loaded {len(transactions)} transactions from file")
    
    # Create analyzer
    analyzer = WalletAnalyzer(
        model_path=args.model,
        data_dir=args.data_dir,
        rpc_url=args.rpc_url,
        device=torch.device(args.device) if args.device else None
    )
    
    # Analyze wallet
    analyzer.analyze_and_print(args.wallet_address, transactions=transactions)


if __name__ == '__main__':
    main()

