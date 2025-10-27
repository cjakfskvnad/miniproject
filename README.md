# DQN Training for Sequence Generation

This project implements DQN training for a Transformer model to generate binary sequences.

## Project Structure

```
miniproject/
├── model/
│   └── mini_transformer.py       # Transformer model definition
├── env/
│   └── sequence_env.py            # Gym environment for sequence generation
├── configs/                       # configs
├── train_dqn.py                   # Main training script
└── README.md                      # This file
```

## Requirements

```bash
pip install torch gymnasium numpy matplotlib tqdm wandb
```




### Train the Model

```bash
python train_dqn.py --config configs/n50.yaml
```

### Find Some Run Logs Here!
https://api.wandb.ai/links/dlwlrma314516-ustc/epp5ugka