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


class SumTree:
    """
    SumTree data structure for efficient prioritized sampling.
    This is a binary tree where each node stores the sum of its children.
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: Maximum number of elements in the tree
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaf nodes
        self.data = np.zeros(capacity, dtype=object)  # Stores actual data
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        """Add a new element with given priority"""
        tree_idx = self.data_pointer + self.capacity - 1

        # Store data
        self.data[self.data_pointer] = data

        # Update tree
        self.update(tree_idx, priority)

        # Increment pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """Update priority of a leaf node and propagate change upward"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        """
        Get leaf index, priority, and data for a given cumulative sum.

        Args:
            s: Cumulative sum value to search for

        Returns:
            leaf_idx: Index in the tree
            priority: Priority value
            data: Stored data
        """
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If we reach bottom, we're done
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            # Descend to left or right child
            if s <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                s -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        """Return total priority (sum of all priorities)"""
        return self.tree[0]

    def max_priority(self):
        """Return maximum priority in the tree"""
        return np.max(self.tree[-self.capacity:])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using SumTree.

    Paper: "Prioritized Experience Replay" by Schaul et al. (2016)
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight (0 = no correction, 1 = full correction)
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # Small constant to avoid zero priority

    def _get_beta(self):
        """Get current beta value (annealed from beta_start to 1.0)"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with maximum priority"""
        # New experiences get maximum priority
        max_priority = self.tree.max_priority()
        if max_priority == 0:
            max_priority = 1.0

        experience = (state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences based on priorities.

        Returns:
            batch: List of experiences
            indices: Indices in the tree (for updating priorities)
            weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []

        # Divide priority range into batch_size segments
        segment = self.tree.total_priority() / batch_size

        beta = self._get_beta()
        self.frame += 1

        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        priorities = np.array(priorities)
        # Probability of sampling = priority / total_priority
        sampling_probabilities = priorities / self.tree.total_priority()

        # Importance sampling weight = (1 / (N * P(i))) ^ beta
        weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)

        # Normalize weights by max weight for stability
        weights = weights / weights.max()

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.

        Args:
            indices: Tree indices to update
            td_errors: TD errors for computing priorities
        """
        for idx, td_error in zip(indices, td_errors):
            # Priority = |TD error| ^ alpha
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        """Return current size of buffer"""
        return self.tree.n_entries



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
        target_update_freq=10,
        use_double_dqn=True,
        use_per=False,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100000
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
            use_double_dqn: Whether to use Double DQN (default: True)
            use_per: Whether to use Prioritized Experience Replay (default: False)
            per_alpha: PER alpha parameter (0 = uniform, 1 = full prioritization)
            per_beta_start: Initial PER beta for importance sampling
            per_beta_frames: Number of frames to anneal beta to 1.0
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

        # Target network (deep copy of q_network)
        import copy
        self.target_network = copy.deepcopy(self.q_network)
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
        self.use_double_dqn = use_double_dqn
        self.use_per = use_per

        # Replay buffer (PER or standard)
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames
            )
        else:
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
        if self.use_per:
            self.replay_buffer.add(state, action, reward, next_state, done)
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        if self.use_per:
            batch, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            batch = random.sample(self.replay_buffer, self.batch_size)
            indices = None
            weights = None

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
            if self.use_double_dqn:
                # Double DQN: use q_network to select action, target_network to evaluate
                next_q_values_online = self.q_network(next_states_tensor)  # (batch, max_seq_len, vocab_size)
                next_q_values_target = self.target_network(next_states_tensor)  # (batch, max_seq_len, vocab_size)

                # Get Q-value for next state using Double DQN
                max_next_q_values = torch.zeros(self.batch_size).to(self.device)
                for i in range(self.batch_size):
                    # Position in next state
                    next_position = len(next_states[i]) - 1
                    # Use q_network to select best action (only consider actions 0 and 1)
                    best_action = torch.argmax(next_q_values_online[i, next_position, :2])
                    # Use target_network to evaluate the Q-value of the selected action
                    max_next_q_values[i] = next_q_values_target[i, next_position, best_action]
            else:
                # Standard DQN: use target_network for both selection and evaluation
                next_q_values = self.target_network(next_states_tensor)  # (batch, max_seq_len, vocab_size)

                # Get max Q-value for next state at its last position
                max_next_q_values = torch.zeros(self.batch_size).to(self.device)
                for i in range(self.batch_size):
                    # Position in next state
                    next_position = len(next_states[i]) - 1
                    # Only consider actions 0 and 1 (not sep token)
                    max_next_q_values[i] = torch.max(next_q_values[i, next_position, :2])

            # Compute target: r + gamma * Q_next(s', a') * (1 - done)
            target_q_values = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)

        # Compute TD errors (for PER priority update)
        td_errors = target_q_values - current_q_values

        # Compute loss (with importance sampling weights if using PER)
        if self.use_per:
            # Element-wise loss weighted by importance sampling weights
            element_wise_loss = torch.nn.functional.mse_loss(current_q_values, target_q_values, reduction='none')
            loss = torch.mean(element_wise_loss * weights)
        else:
            loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities in PER buffer
        if self.use_per:
            td_errors_np = td_errors.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors_np)

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
    eval_freq=100,
    eval_episodes=10,
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
        eval_freq: Evaluate agent every N episodes
        eval_episodes: Number of episodes for each evaluation
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

            # Train agent only if we have at least one batch of data
            if len(agent.replay_buffer) >= agent.batch_size:
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
        if len(agent.replay_buffer) >= agent.batch_size:
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

        # Evaluate agent
        if (episode + 1) % eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating at episode {episode + 1}...")
            print(f"{'='*60}")
            eval_accuracy = evaluate_agent(env, agent, num_episodes=eval_episodes, verbose=False)
            print(f"Evaluation Accuracy: {eval_accuracy:.2%}")

            if use_wandb:
                wandb.log({
                    'episode': episode + 1,
                    'eval_accuracy': eval_accuracy
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
    SEQ_LENGTH = 50
    VOCAB_SIZE = 3
    NUM_EPISODES = 300000
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995
    BUFFER_SIZE = 10000
    BATCH_SIZE = 1024
    TARGET_UPDATE_FREQ = 10
    SAVE_FREQ = 1000000
    LOG_FREQ = 10
    EVAL_FREQ = 50  # Evaluate every 50 episodes
    EVAL_EPISODES = 100  # Number of episodes for each evaluation
    USE_DOUBLE_DQN = True  # Set to False for standard DQN

    # Prioritized Experience Replay settings
    USE_PER = True  # Enable Prioritized Experience Replay
    PER_ALPHA = 0.6  # Prioritization exponent (0 = uniform, 1 = full prioritization)
    PER_BETA_START = 0.4  # Initial importance sampling weight
    PER_BETA_FRAMES = 100000  # Frames to anneal beta to 1.0

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
            "use_double_dqn": USE_DOUBLE_DQN,
            "use_per": USE_PER,
            "per_alpha": PER_ALPHA,
            "per_beta_start": PER_BETA_START,
            "per_beta_frames": PER_BETA_FRAMES,
            "architecture": "MiniTransformer",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "max_seq_len": 128
        },
        name=f"{'ddqn' if USE_DOUBLE_DQN else 'dqn'}{'_per' if USE_PER else ''}-seqlen{SEQ_LENGTH}-ep{NUM_EPISODES}--epsilon{EPSILON_END}",
        tags=["dqn", "transformer", "sequence-generation"] + (["double-dqn"] if USE_DOUBLE_DQN else ["standard-dqn"]) + (["per"] if USE_PER else [])
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
        target_update_freq=TARGET_UPDATE_FREQ,
        use_double_dqn=USE_DOUBLE_DQN,
        use_per=USE_PER,
        per_alpha=PER_ALPHA,
        per_beta_start=PER_BETA_START,
        per_beta_frames=PER_BETA_FRAMES
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
        eval_freq=EVAL_FREQ,
        eval_episodes=EVAL_EPISODES,
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
