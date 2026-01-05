"""
Ethereum Transaction Fraud Detection - Preprocessing and Feature Engineering
This script preprocesses the transaction dataset and creates a comprehensive graph
with rich node and edge features for fraud detection.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, some features will be skipped")


class EthereumGraphPreprocessor:
    """
    Preprocessor for Ethereum transaction data that creates a graph with
    comprehensive node and edge features for fraud detection.
    """
    
    def __init__(self, csv_path, output_dir='./processed_data', 
                 min_degree=1, sample_size=None, random_seed=42):
        """
        Initialize the preprocessor.
        
        Args:
            csv_path: Path to the CSV file containing transactions
            output_dir: Directory to save processed data
            min_degree: Minimum node degree to keep in graph
            sample_size: If provided, randomly sample this many transactions
            random_seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.min_degree = min_degree
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(random_seed)
        
        self.addr2idx = {}
        self.idx2addr = {}
        self.scaler = None
        self.edge_scaler = None
        
    def load_data(self):
        """Load and prepare the transaction dataset."""
        print("Loading transaction data...")
        df = pd.read_csv(self.csv_path)
        
        # Remove unnamed index column if present
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        print(f"Loaded {len(df)} transactions")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Fraud ratio: {df['isError'].sum() / len(df):.4f}")
        
        # Sample if requested (for faster processing)
        if self.sample_size and len(df) > self.sample_size:
            print(f"Sampling {self.sample_size} transactions...")
            df = df.sample(n=self.sample_size, random_state=self.random_seed)
            print(f"Sampled to {len(df)} transactions")
        
        # Clean and convert data types
        df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce').fillna(0).astype(np.int64)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0.0).astype(np.float32)
        df['BlockHeight'] = pd.to_numeric(df['BlockHeight'], errors='coerce').fillna(0).astype(np.int64)
        df['isError'] = pd.to_numeric(df['isError'], errors='coerce').fillna(0).astype(np.int32)
        
        # Ensure addresses are strings
        df['From'] = df['From'].astype(str)
        df['To'] = df['To'].astype(str)
        
        return df
    
    def create_address_mapping(self, df):
        """Create bidirectional mapping between addresses and indices."""
        print("Creating address mapping...")
        
        addresses = pd.concat([df['From'], df['To']]).unique()
        self.addr2idx = {addr: idx for idx, addr in enumerate(addresses)}
        self.idx2addr = {idx: addr for addr, idx in self.addr2idx.items()}
        
        print(f"Unique addresses: {len(self.addr2idx)}")
        return self.addr2idx
    
    def compute_node_features(self, df, cutoff_timestamp=None):
        """
        Compute comprehensive node features including:
        - Transaction statistics (in/out degree, counts, volumes)
        - Temporal features (first/last transaction, duration)
        - Transaction pattern features (unique neighbors, transaction frequency)
        - Fraud exposure, value-weighted entropy, burstiness
        
        Args:
            df: DataFrame with transactions
            cutoff_timestamp: Only use transactions with TimeStamp <= cutoff_timestamp (for temporal splitting)
        """
        print("Computing node features...")
        if cutoff_timestamp is not None:
            print(f"  Using cutoff timestamp: {cutoff_timestamp}")
            df_filtered = df[df['TimeStamp'] <= cutoff_timestamp].copy()
        else:
            df_filtered = df.copy()
        
        n_nodes = len(self.addr2idx)
        
        # Initialize feature arrays
        features = defaultdict(lambda: np.zeros(n_nodes, dtype=np.float32))
        
        # Basic transaction statistics
        features['in_degree'] = np.zeros(n_nodes, dtype=np.float32)
        features['out_degree'] = np.zeros(n_nodes, dtype=np.float32)
        features['total_in_value'] = np.zeros(n_nodes, dtype=np.float32)
        features['total_out_value'] = np.zeros(n_nodes, dtype=np.float32)
        features['avg_in_value'] = np.zeros(n_nodes, dtype=np.float32)
        features['avg_out_value'] = np.zeros(n_nodes, dtype=np.float32)
        features['max_in_value'] = np.zeros(n_nodes, dtype=np.float32)
        features['max_out_value'] = np.zeros(n_nodes, dtype=np.float32)
        
        # Temporal features
        features['first_timestamp'] = np.full(n_nodes, 1e12, dtype=np.float32)
        features['last_timestamp'] = np.zeros(n_nodes, dtype=np.float32)
        features['transaction_span_days'] = np.zeros(n_nodes, dtype=np.float32)
        
        # Error/transaction quality features
        features['error_count'] = np.zeros(n_nodes, dtype=np.float32)
        features['error_rate'] = np.zeros(n_nodes, dtype=np.float32)
        
        # Unique neighbor features
        unique_in_neighbors = [set() for _ in range(n_nodes)]
        unique_out_neighbors = [set() for _ in range(n_nodes)]
        
        # Transaction frequency features
        transaction_times = defaultdict(list)
        
        # Compute features by iterating through transactions
        print("  Processing transactions...")
        for idx, row in df_filtered.iterrows():
            if idx % 50000 == 0:
                print(f"    Processed {idx}/{len(df)} transactions...")
            
            from_addr = row['From']
            to_addr = row['To']
            value = row['Value']
            timestamp = row['TimeStamp']
            is_error = row['isError']
            
            from_idx = self.addr2idx[from_addr]
            to_idx = self.addr2idx[to_addr]
            
            # Update degrees and values
            features['out_degree'][from_idx] += 1
            features['in_degree'][to_idx] += 1
            features['total_out_value'][from_idx] += abs(value)
            features['total_in_value'][to_idx] += abs(value)
            
            # Update max values
            if abs(value) > features['max_out_value'][from_idx]:
                features['max_out_value'][from_idx] = abs(value)
            if abs(value) > features['max_in_value'][to_idx]:
                features['max_in_value'][to_idx] = abs(value)
            
            # Update timestamps
            if timestamp < features['first_timestamp'][from_idx]:
                features['first_timestamp'][from_idx] = timestamp
            if timestamp < features['first_timestamp'][to_idx]:
                features['first_timestamp'][to_idx] = timestamp
            if timestamp > features['last_timestamp'][from_idx]:
                features['last_timestamp'][from_idx] = timestamp
            if timestamp > features['last_timestamp'][to_idx]:
                features['last_timestamp'][to_idx] = timestamp
            
            # Update error counts
            if is_error:
                features['error_count'][from_idx] += 1
                features['error_count'][to_idx] += 1
            
            # Track unique neighbors
            unique_out_neighbors[from_idx].add(to_idx)
            unique_in_neighbors[to_idx].add(from_idx)
            
            # Track transaction times
            transaction_times[from_idx].append(timestamp)
            transaction_times[to_idx].append(timestamp)
        
        # Compute derived features
        print("  Computing derived features...")
        
        # Compute fraud exposure: fraction of neighbors flagged as fraud
        print("  Computing fraud exposure...")
        features['fraud_exposure'] = np.zeros(n_nodes, dtype=np.float32)
        
        # First, identify fraud nodes (nodes that sent fraudulent transactions before cutoff)
        fraud_mask = df_filtered['isError'] == 1
        fraud_senders = df_filtered.loc[fraud_mask, 'From'].unique()
        fraud_node_indices = set()
        for addr in fraud_senders:
            if addr in self.addr2idx:
                fraud_node_indices.add(self.addr2idx[addr])
        
        # For each node, compute fraction of neighbors that are fraud
        for i in range(n_nodes):
            total_neighbors = len(unique_in_neighbors[i]) + len(unique_out_neighbors[i])
            if total_neighbors > 0:
                fraud_neighbors = sum(1 for n in unique_in_neighbors[i] if n in fraud_node_indices)
                fraud_neighbors += sum(1 for n in unique_out_neighbors[i] if n in fraud_node_indices)
                features['fraud_exposure'][i] = fraud_neighbors / total_neighbors
        
        # Compute value-weighted entropy: entropy of outgoing transaction values
        print("  Computing value-weighted entropy...")
        features['value_entropy'] = np.zeros(n_nodes, dtype=np.float32)
        
        # Group outgoing transactions by node
        outgoing_values = defaultdict(list)
        for idx, row in df_filtered.iterrows():
            from_idx = self.addr2idx[row['From']]
            value = abs(row['Value'])
            if value > 0:
                outgoing_values[from_idx].append(value)
        
        # Compute entropy for each node
        for i in range(n_nodes):
            if i in outgoing_values and len(outgoing_values[i]) > 0:
                values = np.array(outgoing_values[i])
                # Normalize to probabilities
                total = values.sum()
                if total > 0:
                    probs = values / total
                    # Compute entropy: -sum(p * log(p))
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    features['value_entropy'][i] = entropy
        
        # Compute burstiness: max_tx_count_in_1h / avg_tx_count
        print("  Computing burstiness...")
        features['burstiness'] = np.zeros(n_nodes, dtype=np.float32)
        
        # Group transactions by node and compute burstiness
        for i in range(n_nodes):
            if i in transaction_times and len(transaction_times[i]) > 1:
                times = np.array(sorted(transaction_times[i]))
                # Convert to hours (assuming Unix timestamp)
                times_hours = times / 3600.0
                
                # Compute max transactions in 1-hour window
                max_tx_in_1h = 0
                window_start = 0
                for window_end in range(len(times_hours)):
                    # Slide window to keep transactions within 1 hour
                    while times_hours[window_end] - times_hours[window_start] > 1.0:
                        window_start += 1
                    max_tx_in_1h = max(max_tx_in_1h, window_end - window_start + 1)
                
                # Average transaction count
                avg_tx_count = len(times) / max(1.0, (times_hours[-1] - times_hours[0]) / 24.0)  # per day
                if avg_tx_count > 0:
                    features['burstiness'][i] = max_tx_in_1h / avg_tx_count
                elif len(times) > 0:
                    # If all transactions in same hour, use count as burstiness
                    features['burstiness'][i] = len(times)
        
        # Average values
        for i in range(n_nodes):
            if features['in_degree'][i] > 0:
                features['avg_in_value'][i] = features['total_in_value'][i] / features['in_degree'][i]
            if features['out_degree'][i] > 0:
                features['avg_out_value'][i] = features['total_out_value'][i] / features['out_degree'][i]
            
            # Error rate
            total_tx = features['in_degree'][i] + features['out_degree'][i]
            if total_tx > 0:
                features['error_rate'][i] = features['error_count'][i] / total_tx
            
            # Transaction span in days (assuming timestamp is Unix timestamp)
            if features['first_timestamp'][i] < 1e11:  # Valid timestamp
                span_seconds = features['last_timestamp'][i] - features['first_timestamp'][i]
                features['transaction_span_days'][i] = max(0, span_seconds / 86400.0)
            
            # Fix invalid timestamps
            if features['first_timestamp'][i] >= 1e11:
                features['first_timestamp'][i] = 0
                features['last_timestamp'][i] = 0
        
        # Unique neighbor counts
        features['unique_in_neighbors'] = np.array([len(unique_in_neighbors[i]) for i in range(n_nodes)], dtype=np.float32)
        features['unique_out_neighbors'] = np.array([len(unique_out_neighbors[i]) for i in range(n_nodes)], dtype=np.float32)
        
        # Transaction frequency features (transactions per day)
        features['tx_frequency'] = np.zeros(n_nodes, dtype=np.float32)
        for i in range(n_nodes):
            if len(transaction_times[i]) > 1 and features['transaction_span_days'][i] > 0:
                features['tx_frequency'][i] = len(transaction_times[i]) / max(features['transaction_span_days'][i], 1.0)
            elif len(transaction_times[i]) > 0:
                features['tx_frequency'][i] = len(transaction_times[i])
        
        # Degree ratio
        total_degree = features['in_degree'] + features['out_degree']
        features['degree_ratio'] = np.divide(
            features['out_degree'], 
            total_degree + 1e-8, 
            out=np.zeros(n_nodes, dtype=np.float32),
            where=(total_degree > 0)
        )
        
        # Value ratio features
        total_value = features['total_in_value'] + features['total_out_value']
        features['value_ratio'] = np.divide(
            features['total_out_value'],
            total_value + 1e-8,
            out=np.zeros(n_nodes, dtype=np.float32),
            where=(total_value > 0)
        )
        
        # Log-scale features for better distribution
        features['log_total_in_value'] = np.log1p(features['total_in_value'])
        features['log_total_out_value'] = np.log1p(features['total_out_value'])
        features['log_avg_in_value'] = np.log1p(features['avg_in_value'])
        features['log_avg_out_value'] = np.log1p(features['avg_out_value'])
        features['log_max_in_value'] = np.log1p(features['max_in_value'])
        features['log_max_out_value'] = np.log1p(features['max_out_value'])
        features['log_transaction_span_days'] = np.log1p(features['transaction_span_days'])
        features['log_value_entropy'] = np.log1p(features['value_entropy'])
        features['log_burstiness'] = np.log1p(features['burstiness'])
        
        return features
    
    def compute_graph_structural_features(self, df, cutoff_timestamp=None):
        """
        Compute graph structural features using NetworkX:
        - PageRank
        - Clustering coefficient
        - Betweenness centrality (sampled for large graphs)
        
        Args:
            df: DataFrame with transactions
            cutoff_timestamp: Only use transactions with TimeStamp <= cutoff_timestamp (for temporal splitting)
        """
        print("Computing graph structural features...")
        if cutoff_timestamp is not None:
            print(f"  Using cutoff timestamp: {cutoff_timestamp}")
            df_filtered = df[df['TimeStamp'] <= cutoff_timestamp].copy()
        else:
            df_filtered = df.copy()
        
        # Build NetworkX graph for structural analysis
        print("  Building NetworkX graph...")
        G = nx.DiGraph()
        
        # Add edges
        for idx, row in df_filtered.iterrows():
            if idx % 50000 == 0:
                print(f"    Added {idx} edges...")
            from_idx = self.addr2idx[row['From']]
            to_idx = self.addr2idx[row['To']]
            G.add_edge(from_idx, to_idx)
        
        print(f"  Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        n_nodes = len(self.addr2idx)
        structural_features = {}
        
        # PageRank
        print("  Computing PageRank...")
        try:
            pagerank = nx.pagerank(G, max_iter=100, tol=1e-6)
            structural_features['pagerank'] = np.array([pagerank.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
            structural_features['log_pagerank'] = np.log1p(structural_features['pagerank'])
        except Exception as e:
            print(f"  Warning: PageRank failed: {e}")
            structural_features['pagerank'] = np.zeros(n_nodes, dtype=np.float32)
            structural_features['log_pagerank'] = np.zeros(n_nodes, dtype=np.float32)
        
        # Clustering coefficient (undirected version)
        print("  Computing clustering coefficient...")
        try:
            G_undirected = G.to_undirected()
            clustering = nx.clustering(G_undirected)
            structural_features['clustering'] = np.array([clustering.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
        except Exception as e:
            print(f"  Warning: Clustering failed: {e}")
            structural_features['clustering'] = np.zeros(n_nodes, dtype=np.float32)
        
        # In-degree and out-degree centrality (normalized)
        print("  Computing degree centralities...")
        try:
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            structural_features['in_degree_centrality'] = np.array([in_degree_centrality.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
            structural_features['out_degree_centrality'] = np.array([out_degree_centrality.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
        except Exception as e:
            print(f"  Warning: Degree centrality failed: {e}")
            structural_features['in_degree_centrality'] = np.zeros(n_nodes, dtype=np.float32)
            structural_features['out_degree_centrality'] = np.zeros(n_nodes, dtype=np.float32)
        
        return structural_features
    
    def compute_edge_features(self, df, cutoff_timestamp=None):
        """
        Compute edge features for each transaction.
        
        Args:
            df: DataFrame with transactions
            cutoff_timestamp: Only use transactions with TimeStamp <= cutoff_timestamp (for temporal splitting)
        """
        print("Computing edge features...")
        if cutoff_timestamp is not None:
            print(f"  Using cutoff timestamp: {cutoff_timestamp}")
            df_filtered = df[df['TimeStamp'] <= cutoff_timestamp].copy()
        else:
            df_filtered = df.copy()
        
        edge_features = []
        
        for idx, row in df_filtered.iterrows():
            if idx % 50000 == 0:
                print(f"  Processed {idx}/{len(df)} edges...")
            
            value = row['Value']
            timestamp = row['TimeStamp']
            block_height = row['BlockHeight']
            is_error = row['isError']
            
            # Edge feature vector
            edge_feat = [
                np.log1p(abs(value)),  # Log value
                timestamp,  # Timestamp
                block_height,  # Block height
                float(is_error),  # Error flag
                np.sign(value) if value != 0 else 0,  # Value direction
            ]
            
            edge_features.append(edge_feat)
        
        return np.array(edge_features, dtype=np.float32)
    
    def create_node_labels(self, df, cutoff_timestamp=None):
        """
        Create node labels from transaction labels.
        
        Args:
            df: DataFrame with transactions
            cutoff_timestamp: Only use transactions with TimeStamp <= cutoff_timestamp (for temporal splitting)
        """
        print("Creating node labels...")
        if cutoff_timestamp is not None:
            print(f"  Using cutoff timestamp: {cutoff_timestamp}")
            df_filtered = df[df['TimeStamp'] <= cutoff_timestamp].copy()
        else:
            df_filtered = df.copy()
        
        n_nodes = len(self.addr2idx)
        node_labels = np.zeros(n_nodes, dtype=np.int64)
        
        # Mark addresses that sent fraudulent transactions
        fraud_mask = df_filtered['isError'] == 1
        fraud_senders = df_filtered.loc[fraud_mask, 'From'].unique()
        
        for addr in fraud_senders:
            if addr in self.addr2idx:
                node_labels[self.addr2idx[addr]] = 1
        
        fraud_count = node_labels.sum()
        print(f"Labeled {fraud_count} fraudulent addresses ({fraud_count/n_nodes*100:.2f}%)")
        
        return node_labels
    
    def build_edge_index(self, df, cutoff_timestamp=None):
        """
        Build edge index for PyTorch Geometric.
        
        Args:
            df: DataFrame with transactions
            cutoff_timestamp: Only use transactions with TimeStamp <= cutoff_timestamp (for temporal splitting)
        """
        print("Building edge index...")
        if cutoff_timestamp is not None:
            print(f"  Using cutoff timestamp: {cutoff_timestamp}")
            df_filtered = df[df['TimeStamp'] <= cutoff_timestamp].copy()
        else:
            df_filtered = df.copy()
        
        src = df_filtered['From'].map(self.addr2idx).to_numpy(dtype=np.int64)
        dst = df_filtered['To'].map(self.addr2idx).to_numpy(dtype=np.int64)
        
        edge_index = np.vstack([src, dst])
        
        return edge_index
    
    def combine_features(self, node_features, structural_features):
        """Combine all node features into a single matrix."""
        print("Combining node features...")
        
        feature_list = []
        feature_names = []
        
        # Add transaction-based features
        tx_feature_names = [
            'in_degree', 'out_degree',
            'log_total_in_value', 'log_total_out_value',
            'log_avg_in_value', 'log_avg_out_value',
            'log_max_in_value', 'log_max_out_value',
            'unique_in_neighbors', 'unique_out_neighbors',
            'transaction_span_days', 'tx_frequency',
            'degree_ratio', 'value_ratio',
            'error_rate',
            'fraud_exposure', 'log_value_entropy', 'log_burstiness'
        ]
        
        for name in tx_feature_names:
            if name in node_features:
                feature_list.append(node_features[name].reshape(-1, 1))
                feature_names.append(name)
        
        # Add structural features
        structural_feature_names = [
            'log_pagerank', 'clustering',
            'in_degree_centrality', 'out_degree_centrality'
        ]
        
        for name in structural_feature_names:
            if name in structural_features:
                feature_list.append(structural_features[name].reshape(-1, 1))
                feature_names.append(name)
        
        # Stack all features
        X = np.hstack(feature_list).astype(np.float32)
        
        print(f"Created feature matrix with shape {X.shape} and {len(feature_names)} features")
        print(f"Features: {', '.join(feature_names)}")
        
        return X, feature_names
    
    def process(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Main processing pipeline with correct temporal splitting.
        
        Uses global cutoff timestamps to build separate graphs for train/val/test.
        Features are recomputed per split using only information available before cutoff.
        
        Args:
            train_ratio: Fraction of transactions for training (default 0.7)
            val_ratio: Fraction of transactions for validation (default 0.15)
            test_ratio: Fraction of transactions for testing (default 0.15)
        """
        print("=" * 60)
        print("Ethereum Transaction Graph Preprocessing")
        print("TEMPORAL SPLITTING: Using global cutoff timestamps")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Create address mapping (using all addresses)
        self.create_address_mapping(df)
        
        # Determine global cutoff timestamps based on transaction timestamps
        print("\nDetermining global cutoff timestamps...")
        valid_timestamps = df['TimeStamp'][df['TimeStamp'] < 1e11].values
        if len(valid_timestamps) == 0:
            raise ValueError("No valid timestamps found in data")
        
        sorted_timestamps = np.sort(valid_timestamps)
        n_total = len(sorted_timestamps)
        
        # Calculate cutoff indices
        train_cutoff_idx = int(n_total * train_ratio)
        val_cutoff_idx = int(n_total * (train_ratio + val_ratio))
        
        # Get cutoff timestamps
        T_train = sorted_timestamps[train_cutoff_idx]
        T_val = sorted_timestamps[val_cutoff_idx]
        T_test = sorted_timestamps[-1]  # Use all data for test
        
        print(f"Train cutoff (T₁): {T_train} ({train_ratio*100:.1f}% of transactions)")
        print(f"Val cutoff (T₂): {T_val} ({(train_ratio+val_ratio)*100:.1f}% of transactions)")
        print(f"Test cutoff (T₃): {T_test} (100% of transactions)")
        
        # Build separate graphs and compute features per split
        print("\n" + "=" * 60)
        print("BUILDING TRAIN GRAPH (transactions <= T₁)")
        print("=" * 60)
        
        # Train split: transactions <= T_train
        train_df = df[df['TimeStamp'] <= T_train].copy()
        print(f"Train transactions: {len(train_df)}")
        
        train_node_features = self.compute_node_features(train_df, cutoff_timestamp=T_train)
        train_structural_features = self.compute_graph_structural_features(train_df, cutoff_timestamp=T_train)
        train_X, feature_names = self.combine_features(train_node_features, train_structural_features)
        train_edge_features = self.compute_edge_features(train_df, cutoff_timestamp=T_train)
        train_edge_index = self.build_edge_index(train_df, cutoff_timestamp=T_train)
        train_labels = self.create_node_labels(train_df, cutoff_timestamp=T_train)
        
        print("\n" + "=" * 60)
        print("BUILDING VAL GRAPH (transactions <= T₂)")
        print("=" * 60)
        
        # Val split: transactions <= T_val
        val_df = df[df['TimeStamp'] <= T_val].copy()
        print(f"Val transactions: {len(val_df)}")
        
        val_node_features = self.compute_node_features(val_df, cutoff_timestamp=T_val)
        val_structural_features = self.compute_graph_structural_features(val_df, cutoff_timestamp=T_val)
        val_X, _ = self.combine_features(val_node_features, val_structural_features)
        val_edge_features = self.compute_edge_features(val_df, cutoff_timestamp=T_val)
        val_edge_index = self.build_edge_index(val_df, cutoff_timestamp=T_val)
        val_labels = self.create_node_labels(val_df, cutoff_timestamp=T_val)
        
        print("\n" + "=" * 60)
        print("BUILDING TEST GRAPH (transactions <= T₃)")
        print("=" * 60)
        
        # Test split: transactions <= T_test (all data)
        test_df = df[df['TimeStamp'] <= T_test].copy()
        print(f"Test transactions: {len(test_df)}")
        
        test_node_features = self.compute_node_features(test_df, cutoff_timestamp=T_test)
        test_structural_features = self.compute_graph_structural_features(test_df, cutoff_timestamp=T_test)
        test_X, _ = self.combine_features(test_node_features, test_structural_features)
        test_edge_features = self.compute_edge_features(test_df, cutoff_timestamp=T_test)
        test_edge_index = self.build_edge_index(test_df, cutoff_timestamp=T_test)
        test_labels = self.create_node_labels(test_df, cutoff_timestamp=T_test)
        
        # Normalize features (fit on train, transform all)
        print("\nNormalizing features (fit on train, transform all)...")
        self.scaler = RobustScaler()
        train_X_scaled = self.scaler.fit_transform(train_X).astype(np.float32)
        val_X_scaled = self.scaler.transform(val_X).astype(np.float32)
        test_X_scaled = self.scaler.transform(test_X).astype(np.float32)
        
        # Normalize edge features (fit on train, transform all)
        print("Normalizing edge features (fit on train, transform all)...")
        self.edge_scaler = RobustScaler()
        train_edge_features_scaled = self.edge_scaler.fit_transform(train_edge_features).astype(np.float32)
        val_edge_features_scaled = self.edge_scaler.transform(val_edge_features).astype(np.float32)
        test_edge_features_scaled = self.edge_scaler.transform(test_edge_features).astype(np.float32)
        
        # Create node masks based on when nodes first appear
        print("\nCreating train/val/test node masks...")
        # A node appears in train if it has transactions <= T_train
        train_nodes = set()
        for addr in train_df['From'].unique():
            if addr in self.addr2idx:
                train_nodes.add(self.addr2idx[addr])
        for addr in train_df['To'].unique():
            if addr in self.addr2idx:
                train_nodes.add(self.addr2idx[addr])
        
        # A node appears in val if it has transactions in (T_train, T_val] but not before T_train
        val_nodes = set()
        val_only_df = df[(df['TimeStamp'] > T_train) & (df['TimeStamp'] <= T_val)]
        for addr in val_only_df['From'].unique():
            if addr in self.addr2idx:
                idx = self.addr2idx[addr]
                if idx not in train_nodes:
                    val_nodes.add(idx)
        for addr in val_only_df['To'].unique():
            if addr in self.addr2idx:
                idx = self.addr2idx[addr]
                if idx not in train_nodes:
                    val_nodes.add(idx)
        
        # A node appears in test if it has transactions in (T_val, T_test] but not before T_val
        test_nodes = set()
        test_only_df = df[(df['TimeStamp'] > T_val) & (df['TimeStamp'] <= T_test)]
        for addr in test_only_df['From'].unique():
            if addr in self.addr2idx:
                idx = self.addr2idx[addr]
                if idx not in train_nodes and idx not in val_nodes:
                    test_nodes.add(idx)
        for addr in test_only_df['To'].unique():
            if addr in self.addr2idx:
                idx = self.addr2idx[addr]
                if idx not in train_nodes and idx not in val_nodes:
                    test_nodes.add(idx)
        
        n_nodes = len(self.addr2idx)
        train_mask = np.array([i in train_nodes for i in range(n_nodes)], dtype=bool)
        val_mask = np.array([i in val_nodes for i in range(n_nodes)], dtype=bool)
        test_mask = np.array([i in test_nodes for i in range(n_nodes)], dtype=bool)
        
        print(f"Train nodes: {train_mask.sum()} ({train_mask.sum()/n_nodes*100:.1f}%)")
        print(f"Val nodes: {val_mask.sum()} ({val_mask.sum()/n_nodes*100:.1f}%)")
        print(f"Test nodes: {test_mask.sum()} ({test_mask.sum()/n_nodes*100:.1f}%)")
        
        # For the final output, we use test graph (most complete) but with proper masks
        # In practice, you might want to use separate graphs, but for compatibility with existing training code,
        # we'll use the test graph structure with proper feature computation per split
        # The key is that features are computed correctly per split
        
        # Use test graph structure (most complete) but features are already computed per split
        # For training, we'll use train_X_scaled, for val use val_X_scaled, for test use test_X_scaled
        # But we need a unified structure, so we'll use test graph and store all feature sets
        
        # Save processed data
        print("\nSaving processed data...")
        output_files = {
            # Use test graph structure (most complete graph)
            'node_features.npy': test_X_scaled,  # Will be replaced per split in training
            'edge_index.npy': test_edge_index,
            'edge_features.npy': test_edge_features_scaled,
            'node_labels.npy': test_labels,
            'train_mask.npy': train_mask,
            'val_mask.npy': val_mask,
            'test_mask.npy': test_mask,
            'addr2idx.pkl': self.addr2idx,
            'idx2addr.pkl': self.idx2addr,
            'feature_names.pkl': feature_names,
            'scaler.pkl': self.scaler,
            'edge_scaler.pkl': self.edge_scaler,
            # Store per-split features
            'train_node_features.npy': train_X_scaled,
            'val_node_features.npy': val_X_scaled,
            'test_node_features.npy': test_X_scaled,
            'train_edge_index.npy': train_edge_index,
            'val_edge_index.npy': val_edge_index,
            'test_edge_index.npy': test_edge_index,
            'train_edge_features.npy': train_edge_features_scaled,
            'val_edge_features.npy': val_edge_features_scaled,
            'test_edge_features.npy': test_edge_features_scaled,
            'train_labels.npy': train_labels,
            'val_labels.npy': val_labels,
            'test_labels.npy': test_labels,
            'cutoff_timestamps.pkl': {
                'T_train': T_train,
                'T_val': T_val,
                'T_test': T_test
            },
            'metadata.pkl': {
                'n_nodes': len(self.addr2idx),
                'n_edges_train': len(train_df),
                'n_edges_val': len(val_df),
                'n_edges_test': len(test_df),
                'n_features': test_X_scaled.shape[1],
                'fraud_ratio_train': train_labels.sum() / len(train_labels) if len(train_labels) > 0 else 0,
                'fraud_ratio_val': val_labels.sum() / len(val_labels) if len(val_labels) > 0 else 0,
                'fraud_ratio_test': test_labels.sum() / len(test_labels) if len(test_labels) > 0 else 0
            }
        }
        
        for filename, data in output_files.items():
            filepath = os.path.join(self.output_dir, filename)
            if filename.endswith('.pkl'):
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                np.save(filepath, data)
            print(f"  Saved {filename}")
        
        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        print("\nIMPORTANT: Features are computed per split using only information before cutoff.")
        print("This prevents temporal data leakage.")
        
        return {
            'node_features': test_X_scaled,
            'edge_index': test_edge_index,
            'edge_features': test_edge_features_scaled,
            'node_labels': test_labels,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'addr2idx': self.addr2idx,
            'idx2addr': self.idx2addr,
            'train_features': train_X_scaled,
            'val_features': val_X_scaled,
            'test_features': test_X_scaled
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Ethereum transaction data')
    parser.add_argument('--csv', type=str, default='Fraud-detection-dataset/first_order_df.csv',
                        help='Path to CSV file')
    parser.add_argument('--output', type=str, default='./processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size (for faster processing, optional)')
    parser.add_argument('--min_degree', type=int, default=1,
                        help='Minimum node degree to keep')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    preprocessor = EthereumGraphPreprocessor(
        csv_path=args.csv,
        output_dir=args.output,
        sample_size=args.sample,
        min_degree=args.min_degree,
        random_seed=args.seed
    )
    
    preprocessor.process()

