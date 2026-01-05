# Quick Start Guide

Get started with Ethereum Fraud Detection in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Preprocess Your Dataset

```bash
python preprocess.py --csv Fraud-detection-dataset/first_order_df.csv --output ./processed_data
```

This will take a few minutes depending on your dataset size. You'll see progress updates as it processes.

**Tip**: For faster processing on large datasets, use `--sample 50000` to test with a subset first.

## Step 3: Train the Model

```bash
python train.py --data_dir ./processed_data --save_dir ./checkpoints
```

Training will automatically:
- Use GPU if available (or CPU otherwise)
- Save the best model based on validation AUC
- Show training progress every 10 epochs
- Generate training curves visualization

**Training time**: Typically 10-30 minutes depending on dataset size and hardware.

## Step 4: Test on a Wallet (Optional)

### Option A: With Web3 (requires RPC URL)

```bash
python test_wallet.py 0xYourWalletAddress \
    --model ./checkpoints/best_model.pt \
    --data_dir ./processed_data \
    --rpc_url https://mainnet.infura.io/v3/YOUR_KEY
```

### Option B: Without Web3 (using CSV)

If you have transactions in CSV format:

```bash
python test_wallet.py 0xYourWalletAddress \
    --model ./checkpoints/best_model.pt \
    --data_dir ./processed_data \
    --transactions_file your_transactions.csv
```

## Expected Output

### Preprocessing
```
============================================================
Ethereum Transaction Graph Preprocessing
============================================================
Loading transaction data...
Loaded 254973 transactions
...
Nodes (unique addresses): 45751
...
Preprocessing complete!
```

### Training
```
Starting Training
============================================================
Epoch  10/200
  Train - Loss: 0.4123, AUC: 0.7234, F1: 0.6543
  Val   - Loss: 0.4234, AUC: 0.7156, F1: 0.6432
...
Best validation AUC: 0.7567 at epoch 45
```

### Testing
```
============================================================
Wallet Fraud Analysis Results
============================================================
Wallet Address: 0x...
Prediction: Fraudulent
Fraud Probability: 0.8234 (82.34%)
Confidence: High
...
```

## Troubleshooting

**Problem**: Out of memory during preprocessing
- **Solution**: Use `--sample 50000` to process a smaller subset

**Problem**: Training is too slow
- **Solution**: Use GPU (`--device cuda`) or reduce model size (`--hidden 64 --heads 4`)

**Problem**: Can't connect to Web3
- **Solution**: Use `--transactions_file` instead, or check your RPC URL

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Experiment with different model architectures (`--model sage`, `--layers 4`)
- Adjust hyperparameters for better performance
- Integrate into your application using the API example in README

## File Structure After Running

```
Pattern-Recognition/
├── processed_data/          # Created by preprocess.py
│   ├── node_features.npy
│   ├── edge_index.npy
│   ├── node_labels.npy
│   └── ...
├── checkpoints/             # Created by train.py
│   ├── best_model.pt
│   ├── training_curves.png
│   └── training_history.pkl
└── ...
```

Happy fraud detecting! 🚀

