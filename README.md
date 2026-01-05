# Ethereum Transaction Fraud Detection with Graph Neural Networks

A comprehensive system for detecting fraudulent Ethereum transactions using Graph Neural Networks (GNNs). This project implements a Temporal Graph Attention Network (TGAT/TGN) with edge-timestamp aware attention that learns from transaction graph structure and temporal patterns to identify fraudulent wallet addresses, similar to what companies like Chainalysis and TRM Labs do.

## Features

- **Comprehensive Feature Engineering**: 
  - Transaction-level features (degree, values, frequencies)
  - Graph structural features (PageRank, clustering, centrality)
  - Temporal features (transaction spans, timestamps)
  - Pattern-based features (unique neighbors, error rates)

- **Advanced GNN Models**: 
  - **Temporal GNN (TGAT/TGN)** - Default model with edge-timestamp aware attention
  - Multi-head Graph Attention Network (GAT)
  - Alternative GraphSAGE implementation
  - Configurable architecture (layers, heads, hidden dimensions)

- **Web3 Integration**: 
  - Fetch wallet transactions using Web3
  - Real-time fraud prediction for any Ethereum address
  - Support for custom RPC endpoints

- **Production-Ready**: 
  - Proper train/val/test splits
  - Early stopping and model checkpointing
  - Comprehensive evaluation metrics
  - Training curves and visualization

## Project Structure

```
Pattern-Recognition/
├── Fraud-detection-dataset/
│   ├── first_order_df.csv          # Main transaction dataset
│   └── second_order_df.csv         # Additional data (optional)
├── preprocess.py                    # Data preprocessing and feature engineering
├── model.py                         # GNN model definitions (TGAT/TGN, GAT, GraphSAGE)
├── train.py                         # Training script
├── test_wallet.py                   # Wallet fraud detection script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### 1. Clone the Repository

```bash
cd Pattern-Recognition
```

### 2. Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; import torch_geometric; print('Installation successful!')"
```

## Dataset

The dataset should be in CSV format with the following columns:

- `TxHash`: Transaction hash (e.g., `0xaca3850ba0080cf47b47f80e46da452f61bcbb5470d3`)
- `BlockHeight`: Block number (e.g., `5848095`)
- `TimeStamp`: Unix timestamp (e.g., `1529873859`)
- `From`: Sender wallet address
- `To`: Receiver wallet address
- `Value`: Transaction value in ETH (e.g., `0.001020`)
- `isError`: Fraud label (0 or 1)

Place your dataset at `Fraud-detection-dataset/first_order_df.csv`.

## Usage

### Step 1: Preprocess the Dataset

Preprocess your transaction data to create a graph with comprehensive features:

```bash
python preprocess.py --csv Fraud-detection-dataset/first_order_df.csv --output ./processed_data
```

**Options:**
- `--csv`: Path to your CSV dataset (default: `Fraud-detection-dataset/first_order_df.csv`)
- `--output`: Output directory for processed data (default: `./processed_data`)
- `--sample`: Optional sample size for faster processing (e.g., `--sample 50000`)
- `--min_degree`: Minimum node degree to keep (default: 1)
- `--seed`: Random seed (default: 42)

**What it does:**
- Creates a transaction graph (nodes = addresses, edges = transactions)
- Computes node features (20+ features including PageRank, clustering, fraud exposure, entropy, burstiness, etc.)
- Computes edge features (value, timestamp, block height, etc.)
- Creates train/val/test splits using **global cutoff timestamps** (prevents temporal data leakage)
- Recomputes features per split using only information available before cutoff time
- Saves processed data for training

**Output files in `processed_data/`:**
- `node_features.npy`: Node feature matrix
- `edge_index.npy`: Graph connectivity
- `edge_features.npy`: Edge feature matrix
- `node_labels.npy`: Fraud labels for nodes
- `train_mask.npy`, `val_mask.npy`, `test_mask.npy`: Data splits
- `scaler.pkl`, `edge_scaler.pkl`: Feature scalers
- `metadata.pkl`: Dataset metadata

### Step 2: Train the Model

Train the GNN model on your processed data:

```bash
python train.py --data_dir ./processed_data --model tgn --save_dir ./checkpoints
```

**Options:**
- `--data_dir`: Directory with processed data (default: `./processed_data`)
- `--model`: Model type - `tgn` (default, Temporal GAT), `tgat` (same as tgn), `gat`, or `sage`
- `--hidden`: Hidden dimension (default: 128)
- `--heads`: Number of attention heads for GAT (default: 8)
- `--layers`: Number of GNN layers (default: 3)
- `--dropout`: Dropout rate (default: 0.5)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Maximum epochs (default: 200)
- `--patience`: Early stopping patience (default: 20)
- `--save_dir`: Checkpoint directory (default: `./checkpoints`)
- `--device`: Device (`cuda` or `cpu`, auto-detect if not specified)

**Example with custom parameters:**

```bash
python train.py \
    --data_dir ./processed_data \
    --model tgn \
    --hidden 256 \
    --heads 8 \
    --layers 4 \
    --dropout 0.3 \
    --lr 0.0005 \
    --epochs 300 \
    --save_dir ./checkpoints
```

**Training output:**
- Progress logs with train/val metrics
- Best model saved to `checkpoints/best_model.pt`
- Training curves saved to `checkpoints/training_curves.png`
- Final test set evaluation with detailed metrics

### Step 3: Test on a Wallet Address

Analyze a wallet address for fraud using the trained model:

#### Option A: Using Web3 (Fetch Transactions Automatically)

```bash
python test_wallet.py 0xYourWalletAddress \
    --model ./checkpoints/best_model.pt \
    --data_dir ./processed_data \
    --rpc_url https://mainnet.infura.io/v3/YOUR_INFURA_KEY
```

#### Option B: Using Pre-fetched Transactions

If you have transactions in a CSV file:

```bash
python test_wallet.py 0xYourWalletAddress \
    --model ./checkpoints/best_model.pt \
    --data_dir ./processed_data \
    --transactions_file path/to/transactions.csv
```

**Options:**
- `wallet_address`: Ethereum wallet address to analyze (required)
- `--model`: Path to trained model (default: `./checkpoints/best_model.pt`)
- `--data_dir`: Directory with processed data (default: `./processed_data`)
- `--rpc_url`: Ethereum RPC URL (Infura, Alchemy, etc.)
- `--transactions_file`: CSV file with transactions (optional)
- `--device`: Device (`cuda` or `cpu`)

**Example RPC URLs:**
- Infura: `https://mainnet.infura.io/v3/YOUR_PROJECT_ID`
- Alchemy: `https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY`
- Public: `https://eth.llamarpc.com` (rate-limited)

**Output:**
```
============================================================
Wallet Fraud Analysis Results
============================================================
Wallet Address: 0x...
Prediction: Fraudulent
Fraud Probability: 0.8234 (82.34%)
Normal Probability: 0.1766 (17.66%)
Confidence: High

Transaction Statistics:
  Total Transactions: 150
  Incoming: 45
  Outgoing: 105
  Total Received: 12.345678 ETH
  Total Sent: 8.765432 ETH
============================================================
```

## Model Architecture

### Temporal Graph Attention Network (TGAT/TGN) - Default Model

The default model incorporates temporal information from edge timestamps into the attention mechanism:

- **Temporal Awareness**: Uses edge timestamps to weight attention (recent transactions get higher attention)
- **Time Encoding**: Sinusoidal encoding of timestamps for temporal representation
- **Edge-Timestamp Integration**: Combines edge features with time encoding
- **Architecture**:
  - Input Layer: Node features (20+ dimensions)
  - Hidden Layers: 3 Temporal GAT layers with 8 attention heads each
  - Output Layer: Binary classification (fraud/normal)
  - Features: 
    - Residual connections
    - Batch normalization
    - Dropout regularization
    - ELU activations
    - Learnable temporal decay

**Why Temporal GNN?**
- Captures time-aware patterns in transaction graphs
- Better handles temporal fraud patterns (e.g., bursty transactions)
- Leverages transaction ordering information
- Particularly effective for detecting time-sensitive fraud behaviors

### Graph Attention Network (GAT)

Standard multi-head attention model:

- **Input Layer**: Node features (20+ dimensions)
- **Hidden Layers**: 3 GAT layers with 8 attention heads each
- **Output Layer**: Binary classification (fraud/normal)
- **Features**: 
  - Residual connections
  - Batch normalization
  - Dropout regularization
  - ELU activations

Can be selected with `--model gat`

### GraphSAGE (Alternative)

A neighborhood aggregation-based model:
- Aggregates information from neighboring nodes
- More efficient for very large graphs
- Can be selected with `--model sage`

## Feature Engineering

The preprocessing pipeline extracts comprehensive features:

### Transaction Features
- In/out degree (number of transactions)
- Total, average, and maximum transaction values
- Transaction frequency (transactions per day)
- Transaction span (first to last transaction)
- Unique neighbors count
- Error rate

### Graph Structural Features
- **PageRank**: Node importance in the graph
- **Clustering Coefficient**: Local clustering measure
- **Degree Centrality**: Normalized degree measures
- **Betweenness Centrality**: Bridge node detection

### Temporal Features
- First and last transaction timestamps
- Transaction duration
- Transaction frequency patterns
- **Fraud exposure**: Fraction of neighbors flagged as fraud
- **Value-weighted entropy**: Entropy of outgoing transaction values (measures laundering behavior)
- **Burstiness**: Max transactions in 1-hour window relative to average (detects bursty fraud patterns)

All features are normalized using RobustScaler (handles outliers better than StandardScaler).

## Performance Tips

1. **Large Datasets**: Use `--sample` in preprocessing to work with a subset first
2. **GPU Training**: Model automatically uses GPU if available (`--device cuda`)
3. **Memory Issues**: Reduce `--hidden` and `--heads` if running out of memory
4. **Faster Training**: Reduce `--layers` or use GraphSAGE (`--model sage`)
5. **Better Performance**: 
   - Use Temporal GNN (`--model tgn`) for time-aware patterns (default)
   - Increase model capacity (`--hidden 256 --heads 8`)
   - Temporal GNN leverages edge timestamps for better fraud detection

## Troubleshooting

### Issue: "Out of Memory" during training
**Solution**: Reduce model size or use a smaller sample
```bash
python train.py --hidden 64 --heads 4 --layers 2
```

### Issue: Web3 connection fails
**Solution**: Check your RPC URL or use transactions file instead
- Verify RPC URL is correct
- Check API key is valid
- Try a different RPC provider
- Use `--transactions_file` to skip Web3

### Issue: Low model performance
**Solution**: 
- Check fraud ratio in dataset (should have reasonable balance)
- Try increasing model capacity
- Adjust learning rate (`--lr 0.0005`)
- Train for more epochs (`--epochs 300`)

### Issue: Preprocessing takes too long
**Solution**: 
- Use `--sample` to process a subset
- Skip structural features (modify preprocess.py)
- Process in batches

## API Integration Example

For production use, you can integrate the model into your application:

```python
from test_wallet import WalletAnalyzer
import torch

# Initialize analyzer
analyzer = WalletAnalyzer(
    model_path='./checkpoints/best_model.pt',
    data_dir='./processed_data',
    rpc_url='https://mainnet.infura.io/v3/YOUR_KEY'
)

# Analyze wallet
result = analyzer.predict_wallet('0xYourWalletAddress')
print(f"Fraud probability: {result['fraud_probability']}")
```

## Evaluation Metrics

The model is evaluated using:
- **AUC-ROC**: Area under the ROC curve (primary metric)
- **Accuracy**: Overall classification accuracy
- **Precision**: Fraud detection precision
- **Recall**: Fraud detection recall
- **F1-Score**: Harmonic mean of precision and recall

Class-weighted loss is used to handle imbalanced datasets.

## Future Enhancements

- [x] Temporal GNN for time-series patterns (✅ Implemented as TGAT/TGN)
- [ ] Multi-hop subgraph extraction for wallet analysis
- [ ] Integration with blockchain explorer APIs
- [ ] Real-time transaction monitoring
- [ ] Explainability/attention visualization
- [ ] Multi-chain support (BSC, Polygon, etc.)

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- PyTorch
- PyTorch Geometric
- NetworkX
- scikit-learn
- pandas, numpy
- web3 (for wallet analysis)

## License

This project is for educational and research purposes.

## Acknowledgments

- PyTorch Geometric team for the excellent GNN framework
- Ethereum community for blockchain data
- Chainalysis and TRM Labs for inspiration on fraud detection approaches

## Contact

For questions or issues, please open an issue on the repository.

---

**Note**: This is a research tool. Always verify predictions with additional analysis before making financial or security decisions.

