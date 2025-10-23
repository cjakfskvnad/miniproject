# DQN Training for Sequence Generation

This project implements DQN training for a Transformer model to generate binary sequences.

## Project Structure

```
miniproject/
├── model/
│   └── mini_transformer.py       # Transformer model definition
├── env/
│   └── sequence_env.py            # Gym environment for sequence generation
├── utils/
│   └── training_utils.py          # Utility functions and visualization
├── train_dqn.py                   # Main training script
└── README.md                      # This file
```

## Requirements

```bash
pip install torch gymnasium numpy matplotlib tqdm
```

## Quick Start

### 1. Test the Environment

```bash
python env/sequence_env.py
```

This will run a test of the environment with random actions.

### 2. Train the Model

```bash
python train_dqn.py
```

This will:
- Create a sequence generation environment
- Initialize a DQN agent with a Transformer model
- Train for 1000 episodes
- Save checkpoints every 100 episodes to `checkpoints/`
- Evaluate the trained agent

### 3. Monitor Training

Training progress is logged every 10 episodes, showing:
- Average reward
- Average accuracy
- Average loss
- Epsilon value
- Replay buffer size

## Configuration

You can modify hyperparameters in `train_dqn.py`:

```python
SEQ_LENGTH = 10              # Length of target sequences
NUM_EPISODES = 1000          # Number of training episodes
LEARNING_RATE = 1e-4         # Learning rate
GAMMA = 0.99                 # Discount factor
EPSILON_START = 1.0          # Initial exploration rate
EPSILON_END = 0.01           # Minimum exploration rate
EPSILON_DECAY = 0.995        # Epsilon decay rate
BUFFER_SIZE = 10000          # Replay buffer size
BATCH_SIZE = 32              # Training batch size
TARGET_UPDATE_FREQ = 10      # Target network update frequency
```

## How It Works

### Environment (`sequence_env.py`)

The environment generates random binary sequences (0s and 1s) and rewards the agent for correctly predicting each token:

- **State**: Current sequence (with sep token at start)
- **Action**: Generate token 0 or 1
- **Reward**: +1 if predicted token matches target at that position, 0 otherwise
- **Episode**: Complete when full sequence is generated

### DQN Agent (`train_dqn.py`)

The agent uses:
- **Q-Network**: Transformer model that predicts Q-values for actions
- **Target Network**: Stabilizes training
- **Experience Replay**: Learns from past experiences
- **Epsilon-Greedy**: Balances exploration and exploitation

### Training Process

1. Agent observes current sequence state
2. Selects action (0 or 1) using epsilon-greedy policy
3. Receives reward based on whether action matches target
4. Stores experience in replay buffer
5. Samples batch from buffer and updates Q-network
6. Periodically updates target network
7. Decays epsilon over time

## Output

### Checkpoints

Models are saved to `checkpoints/`:
- `agent_episode_100.pt`, `agent_episode_200.pt`, etc.
- `agent_final.pt` (final trained model)

### Evaluation

After training, the agent is evaluated on 10 episodes showing:
- Target sequence
- Generated sequence
- Accuracy per episode
- Average accuracy

## Example Output

```
Episode 1000/1000
  Avg Reward: 8.50/10
  Avg Accuracy: 85.00%
  Avg Loss: 0.0234
  Epsilon: 0.0100
  Buffer Size: 10000

Evaluating trained agent...
Episode 1: Reward = 9/10, Accuracy = 90.00%
  Target:    [1 0 1 1 0 0 1 0 1 0]
  Generated: [1 0 1 1 0 1 1 0 1 0]

Average Accuracy: 87.00%
```

## Advanced Usage

### Load and Continue Training

```python
from train_dqn import DQNAgent, train_dqn
from env.sequence_env import SequenceGenerationEnv
from model.mini_transformer import MiniTransformer

# Create environment and model
env = SequenceGenerationEnv(seq_length=10)
model = MiniTransformer(vocab_size=3, d_model=64, nhead=4, num_layers=2)

# Create agent and load checkpoint
agent = DQNAgent(model=model)
agent.load('checkpoints/agent_episode_500.pt')

# Continue training
train_dqn(env, agent, num_episodes=500)
```

### Evaluate Only

```python
from train_dqn import DQNAgent, evaluate_agent
from env.sequence_env import SequenceGenerationEnv
from model.mini_transformer import MiniTransformer

env = SequenceGenerationEnv(seq_length=10)
model = MiniTransformer(vocab_size=3, d_model=64, nhead=4, num_layers=2)

agent = DQNAgent(model=model)
agent.load('checkpoints/agent_final.pt')

evaluate_agent(env, agent, num_episodes=20, verbose=True)
```

### Generate Sequences

```python
from utils.training_utils import generate_sequence_with_model
import torch

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Greedy generation
sequence = generate_sequence_with_model(model, seq_length=10, device=device, greedy=True)
print(f"Generated: {sequence}")

# Sampling with temperature
sequence = generate_sequence_with_model(model, seq_length=10, device=device, greedy=False, temperature=0.8)
print(f"Sampled: {sequence}")
```

## Notes

- The model learns to predict binary sequences token by token
- Training typically converges to 80-95% accuracy depending on sequence length
- Longer sequences are harder to learn
- Adjust hyperparameters for different sequence lengths
- GPU training is significantly faster if available

## Troubleshooting

**Low accuracy after training:**
- Increase number of episodes
- Adjust learning rate
- Increase model size (d_model, num_layers)
- Decrease epsilon decay rate for more exploration

**Training is slow:**
- Reduce buffer size
- Reduce batch size
- Use GPU if available
- Decrease sequence length for initial testing

**Loss not decreasing:**
- Check learning rate (try 1e-3 or 1e-5)
- Ensure replay buffer is filling up
- Verify target network is updating
- Check gradient clipping threshold
