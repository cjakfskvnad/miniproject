import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from tqdm import tqdm
import wandb

from env.sequence_env import SequenceGenerationEnv
from model.mini_transformer import MiniTransformer

import debugpy

# Enable debugpy for remote debugging
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached!")

class DQNAgent:
    """
    DQN Agent that uses a Transformer model as the Q-network.
    """

    def __init__(
        self,
        model,
        vocab_size=3,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=10
    ):
        """
        Args:
            model: MiniTransformer model
            vocab_size: Size of vocabulary
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network (in episodes)
        """
        # Use MPS (Metal Performance Shaders) for Mac, CUDA for NVIDIA, else CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Q-network (policy network)
        self.q_network = model.to(self.device)

        # Target network
        self.target_network = MiniTransformer(
            vocab_size=vocab_size,
            d_model=model.d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_seq_len=model.max_seq_len
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Training stats
        self.episode_count = 0
        self.train_step_count = 0

    def select_action(self, sequence, epsilon=None):
        """
        Select action using epsilon-greedy policy.

        Args:
            sequence: Current sequence tensor (1, seq_len)
            epsilon: Override epsilon value (optional)

        Returns:
            action: 0 or 1
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            # Explore: random action (0 or 1)
            return random.randint(0, 1)
        else:
            # Exploit: choose action with highest Q-value
            with torch.no_grad():
                sequence = sequence.to(self.device)
                q_values = self.q_network(sequence)  # (1, seq_len, vocab_size)

                # Get Q-values for the last position (where we're making decision)
                # vocab: {0: binary_0, 1: binary_1, 2: sep_token}
                # We only consider actions 0 and 1 (binary tokens)
                last_q_values = q_values[0, -1, :2]  # Only tokens 0 and 1

                action = torch.argmax(last_q_values).item()
                return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Prepare batch data (states are now numpy arrays of fixed length)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Pad states and next_states to the same length
        max_state_len = max(len(s) for s in states)
        max_next_state_len = max(len(s) for s in next_states)

        # Pad states with 0s
        padded_states = np.zeros((self.batch_size, max_state_len), dtype=np.int32)
        for i, state in enumerate(states):
            padded_states[i, :len(state)] = state

        # Pad next_states with 0s
        padded_next_states = np.zeros((self.batch_size, max_next_state_len), dtype=np.int32)
        for i, next_state in enumerate(next_states):
            padded_next_states[i, :len(next_state)] = next_state

        # Convert to tensors
        states_tensor = torch.tensor(padded_states, dtype=torch.long).to(self.device)
        next_states_tensor = torch.tensor(padded_next_states, dtype=torch.long).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute current Q-values
        q_values = self.q_network(states_tensor)  # (batch, max_seq_len, vocab_size)

        # Get Q-values at the current position (last token position) for selected actions
        # State format: [target, sep, generated...]
        # The position where decision was made is len(state) - 1
        current_q_values = torch.zeros(self.batch_size).to(self.device)
        for i in range(self.batch_size):
            # Position in the sequence where action was taken
            position = len(states[i]) - 1
            current_q_values[i] = q_values[i, position, actions_tensor[i]]

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)  # (batch, max_seq_len, vocab_size)

            # Get max Q-value for next state at its last position
            max_next_q_values = torch.zeros(self.batch_size).to(self.device)
            for i in range(self.batch_size):
                # Position in next state
                next_position = len(next_states[i]) - 1
                # Only consider actions 0 and 1 (not sep token)
                max_next_q_values[i] = torch.max(next_q_values[i, next_position, :2])

            # Compute target: r + gamma * max_Q(s', a') * (1 - done)
            target_q_values = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'train_step_count': self.train_step_count
        }, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.train_step_count = checkpoint['train_step_count']
        print(f"Agent loaded from {filepath}")


def train_dqn(
    env: SequenceGenerationEnv,
    agent :DQNAgent,
    num_episodes=1000,
    max_steps_per_episode=None,
    save_dir='checkpoints',
    save_freq=100,
    log_freq=10,
    use_wandb=True
):
    """
    Train DQN agent.

    Args:
        env: Gym environment
        agent: DQN agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode (None = env default)
        save_dir: Directory to save checkpoints
        save_freq: Save checkpoint every N episodes
        log_freq: Log statistics every N episodes
        use_wandb: Whether to use wandb for logging
    """
    os.makedirs(save_dir, exist_ok=True)

    episode_rewards = []
    episode_accuracies = []
    episode_losses = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        state, info = env.reset()  # state is now [target, sep, generated]

        episode_reward = 0
        episode_loss_sum = 0
        episode_loss_count = 0
        terminated = False
        step_count = 0

        while not terminated:
            # Get current state tensor (already contains target + sep + generated)
            state_tensor = torch.tensor([state], dtype=torch.long)

            # Select action
            action = agent.select_action(state_tensor)

            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, terminated)

            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss_sum += loss
                episode_loss_count += 1

            # Update state
            state = next_state
            episode_reward += reward
            step_count += 1

            # Check if max steps reached
            if max_steps_per_episode and step_count >= max_steps_per_episode:
                break

        # Update target network periodically
        agent.episode_count += 1
        if agent.episode_count % agent.target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.update_epsilon()

        # Track statistics
        episode_rewards.append(episode_reward)
        accuracy = episode_reward / env.seq_length
        episode_accuracies.append(accuracy)

        avg_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0
        episode_losses.append(avg_loss)

        # Log to wandb
        if use_wandb:
            wandb.log({
                'episode': episode + 1,
                'episode_reward': episode_reward,
                'episode_accuracy': accuracy,
                'episode_loss': avg_loss,
                'epsilon': agent.epsilon,
                'buffer_size': len(agent.replay_buffer),
                'steps': step_count
            })

        # Log progress
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_accuracy = np.mean(episode_accuracies[-log_freq:])
            avg_loss = np.mean(episode_losses[-log_freq:])

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}/{env.seq_length}")
            print(f"  Avg Accuracy: {avg_accuracy:.2%}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")

            # Log aggregated metrics to wandb
            if use_wandb:
                wandb.log({
                    f'avg_{log_freq}_reward': avg_reward,
                    f'avg_{log_freq}_accuracy': avg_accuracy,
                    f'avg_{log_freq}_loss': avg_loss
                })

        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            save_path = os.path.join(save_dir, f'agent_episode_{episode + 1}.pt')
            agent.save(save_path)

    # Save final model
    final_save_path = os.path.join(save_dir, 'agent_final.pt')
    agent.save(final_save_path)

    return episode_rewards, episode_accuracies, episode_losses


def evaluate_agent(env, agent, num_episodes=10, verbose=True):
    """
    Evaluate trained agent.

    Args:
        env: Gym environment
        agent: Trained DQN agent
        num_episodes: Number of evaluation episodes
        verbose: Print detailed results

    Returns:
        average accuracy
    """
    agent.q_network.eval()

    total_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()

        episode_reward = 0
        terminated = False

        with torch.no_grad():
            while not terminated:
                state_tensor = torch.tensor([state], dtype=torch.long)
                action = agent.select_action(state_tensor, epsilon=0.0)  # Greedy
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

        total_rewards.append(episode_reward)

        if verbose:
            print(f"Episode {episode + 1}: Reward = {episode_reward}/{env.seq_length}, "
                  f"Accuracy = {episode_reward/env.seq_length:.2%}")
            print(f"  Target:    {info['target_sequence']}")
            print(f"  Generated: {info['generated_sequence']}")

    avg_accuracy = np.mean(total_rewards) / env.seq_length
    print(f"\nAverage Accuracy: {avg_accuracy:.2%}")

    agent.q_network.train()
    return avg_accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("DQN Training for Sequence Generation")
    print("=" * 60)

    # Configuration
    SEQ_LENGTH = 10
    VOCAB_SIZE = 3
    NUM_EPISODES = 1000
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_SIZE = 10000
    BATCH_SIZE = 1024
    TARGET_UPDATE_FREQ = 10
    SAVE_FREQ = 100
    LOG_FREQ = 10

    # Initialize wandb
    wandb.init(
        project="dqn-sequence-generation",
        config={
            "seq_length": SEQ_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "num_episodes": NUM_EPISODES,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "target_update_freq": TARGET_UPDATE_FREQ,
            "architecture": "MiniTransformer",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "max_seq_len": 128
        },
        name=f"dqn-seqlen{SEQ_LENGTH}-ep{NUM_EPISODES}",
        tags=["dqn", "transformer", "sequence-generation"]
    )

    # Create environment
    print(f"\nCreating environment (sequence length = {SEQ_LENGTH})...")
    env = SequenceGenerationEnv(seq_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE)

    # Create model
    print("Creating Transformer model...")
    model = MiniTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_seq_len=128
    )

    # Create agent
    print("Creating DQN agent...")
    agent = DQNAgent(
        model=model,
        vocab_size=VOCAB_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    total_params = sum(p.numel() for p in agent.q_network.parameters())
    print(f"\nTotal parameters: {total_params}")
    print(f"Device: {agent.device}")



    # Train
    print(f"\nStarting training for {NUM_EPISODES} episodes...")
    print("=" * 60)

    rewards, accuracies, losses = train_dqn(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        save_dir='checkpoints',
        save_freq=SAVE_FREQ,
        log_freq=LOG_FREQ,
        use_wandb=True
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating trained agent...")
    print("=" * 60)
    final_accuracy = evaluate_agent(env, agent, num_episodes=10, verbose=True)

    # Log final evaluation results to wandb
    wandb.log({"final_evaluation_accuracy": final_accuracy})


    # Finish wandb run
    wandb.finish()

    print("\nTraining complete!")
