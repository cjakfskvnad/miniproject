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
        per_beta_frames=100000,
        n_step=3
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
            n_step: Number of steps for n-step returns (1 = standard TD)
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
        self.n_step = n_step

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

        # N-step buffer for computing n-step returns
        self.n_step_buffer = deque(maxlen=n_step)

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

    def select_actions_batch(self, states, epsilon=None):
        """
        Select actions for a batch of states using epsilon-greedy policy.
        Utilizes transformer's batch inference capability for efficiency.

        Args:
            states: List of state arrays (each state can have different length)
            epsilon: Override epsilon value (optional)

        Returns:
            actions: List of actions (0 or 1) for each state
        """
        if epsilon is None:
            epsilon = self.epsilon

        batch_size = len(states)
        actions = []

        # Determine which states will explore vs exploit
        explore_mask = [random.random() < epsilon for _ in range(batch_size)]

        # For exploring states, generate random actions
        for i in range(batch_size):
            if explore_mask[i]:
                actions.append(random.randint(0, 1))
            else:
                actions.append(None)  # Placeholder for exploit actions

        # If all states are exploring, return early
        if all(explore_mask):
            return actions

        # Get indices of states that need exploitation
        exploit_indices = [i for i, explore in enumerate(explore_mask) if not explore]

        if len(exploit_indices) > 0:
            # Collect states that need exploitation
            exploit_states = [states[i] for i in exploit_indices]

            # Pad states to same length
            max_len = max(len(s) for s in exploit_states)
            padded_states = np.zeros((len(exploit_states), max_len), dtype=np.int32)
            state_lengths = []

            for i, state in enumerate(exploit_states):
                padded_states[i, :len(state)] = state
                state_lengths.append(len(state))

            # Batch forward pass
            with torch.no_grad():
                states_tensor = torch.tensor(padded_states, dtype=torch.long).to(self.device)
                q_values = self.q_network(states_tensor)  # (batch, max_len, vocab_size)

                # Extract Q-values at the last position for each state
                for i, state_len in enumerate(state_lengths):
                    # Get Q-values at the last valid position
                    last_q_values = q_values[i, state_len - 1, :2]  # Only actions 0 and 1
                    action = torch.argmax(last_q_values).item()

                    # Place action in the correct position
                    original_idx = exploit_indices[i]
                    actions[original_idx] = action

        return actions

    def _get_n_step_info(self):
        """
        Calculate n-step return and get n-step transition.

        Returns:
            state: Initial state
            action: Initial action
            n_step_reward: Sum of discounted rewards
            next_state: State after n steps
            done: Whether episode ended
        """
        # Get the first transition
        state, action = self.n_step_buffer[0][:2]

        # Calculate n-step discounted reward
        n_step_reward = 0
        for i, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * reward

        # Get the last transition's next_state and done
        _, _, _, next_state, done = self.n_step_buffer[-1]

        return state, action, n_step_reward, next_state, done

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer using n-step returns"""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Only store in main buffer when we have n transitions or episode ends
        if len(self.n_step_buffer) == self.n_step or done:
            # Get n-step transition
            n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info()

            # Store in main replay buffer
            if self.use_per:
                self.replay_buffer.add(n_state, n_action, n_reward, n_next_state, n_done)
            else:
                self.replay_buffer.append((n_state, n_action, n_reward, n_next_state, n_done))

        # If episode ends, flush remaining transitions in n_step_buffer
        if done:
            # Store all remaining transitions with their respective n-step returns
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                if len(self.n_step_buffer) > 0:
                    n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info()
                    if self.use_per:
                        self.replay_buffer.add(n_state, n_action, n_reward, n_next_state, n_done)
                    else:
                        self.replay_buffer.append((n_state, n_action, n_reward, n_next_state, n_done))

            # Clear n-step buffer for next episode
            self.n_step_buffer.clear()

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
    agent: DQNAgent,
    num_episodes=1000,
    max_steps_per_episode=None,
    save_dir='checkpoints',
    save_freq=100,
    log_freq=10,
    eval_freq=100,
    eval_episodes=10,
    use_wandb=True,
    env_batch_size=32,
    train_steps_per_iter=1
):
    """
    Train DQN agent using batch environments.

    Args:
        env: Template Gym environment (will be used to create multiple environments)
        agent: DQN agent
        num_episodes: Total number of episodes to train
        max_steps_per_episode: Maximum steps per episode (None = env default)
        save_dir: Directory to save checkpoints
        save_freq: Save checkpoint every N episodes
        log_freq: Log statistics every N episodes
        eval_freq: Evaluate agent every N episodes
        eval_episodes: Number of episodes for each evaluation
        use_wandb: Whether to use wandb for logging
        env_batch_size: Number of parallel environments to run
        train_steps_per_iter: Number of training steps per iteration
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create batch of environments
    envs = [SequenceGenerationEnv(seq_length=env.seq_length, vocab_size=env.vocab_size, sep_token=env.sep_token)
            for _ in range(env_batch_size)]

    # Initialize all environments
    states = []
    for env_idx in range(env_batch_size):
        state, info = envs[env_idx].reset()
        states.append(state)

    # Track statistics for each environment
    episode_rewards = []  # Global episode rewards
    episode_accuracies = []  # Global episode accuracies
    episode_losses = []  # Track losses

    # Per-environment tracking
    env_episode_rewards = [0.0] * env_batch_size
    env_step_counts = [0] * env_batch_size
    completed_episodes = 0

    # Training loop
    global_step = 0
    episode_loss_sum = 0
    episode_loss_count = 0

    pbar = tqdm(total=num_episodes, desc="Training")

    while completed_episodes < num_episodes:
        # Select actions for all environments in batch
        actions = agent.select_actions_batch(states)

        # Step all environments and collect transitions
        next_states = []
        for env_idx in range(env_batch_size):
            # Take action in environment
            next_state, reward, terminated, truncated, info = envs[env_idx].step(actions[env_idx])
            next_states.append(next_state)

            # Store transition
            agent.store_transition(states[env_idx], actions[env_idx], reward, next_state, terminated)

            # Update per-environment statistics
            env_episode_rewards[env_idx] += reward
            env_step_counts[env_idx] += 1

            # Check if episode ended or max steps reached
            should_reset = terminated
            if max_steps_per_episode and env_step_counts[env_idx] >= max_steps_per_episode:
                should_reset = True

            # If environment finished, log and reset
            if should_reset:
                completed_episodes += 1

                # Log episode statistics
                episode_reward = env_episode_rewards[env_idx]
                accuracy = episode_reward / envs[env_idx].seq_length
                episode_rewards.append(episode_reward)
                episode_accuracies.append(accuracy)

                # Log to wandb
                if use_wandb:
                    wandb.log({
                        'episode': completed_episodes,
                        'episode_reward': episode_reward,
                        'episode_accuracy': accuracy,
                        'epsilon': agent.epsilon,
                        'buffer_size': len(agent.replay_buffer),
                        'steps': env_step_counts[env_idx]
                    }, step=global_step)

                # Update progress bar
                pbar.update(1)

                # Reset environment and statistics
                next_state, info = envs[env_idx].reset()
                next_states[env_idx] = next_state
                env_episode_rewards[env_idx] = 0.0
                env_step_counts[env_idx] = 0

                # Update target network periodically
                agent.episode_count += 1
                # if agent.episode_count % agent.target_update_freq == 0:
                #     agent.update_target_network()

                # Decay epsilon
                if completed_episodes > 0 and completed_episodes % log_freq == 0 and completed_episodes // log_freq != (completed_episodes - 1) // log_freq:
                    # Calculate averages over last log_freq episodes
                    recent_start = max(0, len(episode_rewards) - log_freq)
                    avg_reward = np.mean(episode_rewards[recent_start:])
                    avg_accuracy = np.mean(episode_accuracies[recent_start:])

                    # Calculate average loss
                    recent_loss_start = max(0, len(episode_losses) - 100)
                    avg_loss = np.mean(episode_losses[recent_loss_start:]) if len(episode_losses) > 0 else 0

                    print(f"\nEpisode {completed_episodes}/{num_episodes}")
                    print(f"  Avg Reward: {avg_reward:.2f}/{env.seq_length}")
                    print(f"  Avg Accuracy: {avg_accuracy:.2%}")
                    print(f"  Avg Loss: {avg_loss:.4f}")
                    print(f"  Epsilon: {agent.epsilon:.4f}")
                    print(f"  Buffer Size: {len(agent.replay_buffer)}")

                    # Log aggregated metrics to wandb
                    if use_wandb:
                        wandb.log({
                            'episode': completed_episodes,
                            f'avg_{log_freq}_reward': avg_reward,
                            f'avg_{log_freq}_accuracy': avg_accuracy,
                            f'avg_{log_freq}_loss': avg_loss
                        }, step=global_step)
                if completed_episodes > 0 and completed_episodes % save_freq == 0 and completed_episodes // save_freq != (completed_episodes - 1) // save_freq:
                    save_path = os.path.join(save_dir, f'agent_episode_{completed_episodes}.pt')
                    agent.save(save_path)

        # Evaluate agent (based on global_step)
        if global_step > 0 and global_step % eval_freq == 0 and global_step // eval_freq != (global_step - 1) // eval_freq:
            print(f"\n{'='*60}")
            print(f"Evaluating at global step {global_step}...")
            print(f"{'='*60}")
            eval_accuracy = evaluate_agent(env, agent, num_episodes=eval_episodes, verbose=False)
            print(f"Evaluation Accuracy: {eval_accuracy:.2%}")

            if use_wandb:
                wandb.log({
                    'eval_accuracy': eval_accuracy
                }, step=global_step)

        # Train agent multiple times per batch step
        if len(agent.replay_buffer) >= agent.batch_size:
            for _ in range(train_steps_per_iter):
                loss = agent.train_step()
                if loss is not None:
                    episode_loss_sum += loss
                    episode_loss_count += 1
                    episode_losses.append(loss)

        # Update states for next iteration
        states = next_states
        global_step += 1
        if global_step % agent.target_update_freq == 0:
            agent.update_target_network()
        if len(agent.replay_buffer) >= agent.batch_size:
            agent.update_epsilon()


        # Log progress periodically


        # Save checkpoint


    pbar.close()

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
    NUM_EPISODES = 300000000
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    BUFFER_SIZE = 10000
    BATCH_SIZE = 1024
    
    SAVE_FREQ = 10000
    LOG_FREQ = 10
    EVAL_FREQ = 50  # Evaluate every 50 episodes
    EVAL_EPISODES = 100 
     # Number of episodes for each evaluation

    USE_DOUBLE_DQN = True  # Set to False for standard DQN
    TARGET_UPDATE_FREQ = 100
    ENV_BATCH_SIZE = 1  # Number of parallel environments
    TRAIN_STEPS_PER_ITER = 1  
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995# Number of training steps per iteration

    # Prioritized Experience Replay settings
    USE_PER = True  # Enable Prioritized Experience Replay
    PER_ALPHA = 0.6  # Prioritization exponent (0 = uniform, 1 = full prioritization)
    PER_BETA_START = 0.4  # Initial importance sampling weight
    PER_BETA_FRAMES = 100000  # Frames to anneal beta to 1.0

    NUM_LAYER = 8
    NHEAD = 8

    # N-step learning settings
    N_STEP = 1  # Number of steps for n-step returns (1 = standard TD, 3-5 recommended)

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
            "n_step": N_STEP,
            "env_batch_size": ENV_BATCH_SIZE,
            "train_steps_per_iter": TRAIN_STEPS_PER_ITER,
            "architecture": "MiniTransformer",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "max_seq_len": 128
        },
        name=f"{'ddqn' if USE_DOUBLE_DQN else 'dqn'}{'_per' if USE_PER else ''}_n{N_STEP}-seqlen{SEQ_LENGTH}-ep{NUM_EPISODES}--epsilon{EPSILON_END}--update{TARGET_UPDATE_FREQ}--envBATCH{ENV_BATCH_SIZE}--trainstep{TRAIN_STEPS_PER_ITER}---numlayer{NUM_LAYER}--nhead{NHEAD}",
        tags=["dqn", "transformer", "sequence-generation"] + (["double-dqn"] if USE_DOUBLE_DQN else ["standard-dqn"]) + (["per"] if USE_PER else []) + ([f"n-step-{N_STEP}"] if N_STEP > 1 else [])
    )

    # Create save directory based on wandb run name
    save_dir = os.path.join('checkpoints', wandb.run.name)
    print(f"\nCheckpoints will be saved to: {save_dir}")

    # Create environment
    print(f"\nCreating environment (sequence length = {SEQ_LENGTH})...")
    env = SequenceGenerationEnv(seq_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE)

    # Create model
    print("Creating Transformer model...")
    model = MiniTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        nhead=NHEAD,
        num_layers=NUM_LAYER,
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
        per_beta_frames=PER_BETA_FRAMES,
        n_step=N_STEP
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
        save_dir=save_dir,
        save_freq=SAVE_FREQ,
        log_freq=LOG_FREQ,
        eval_freq=EVAL_FREQ,
        eval_episodes=EVAL_EPISODES,
        use_wandb=True,
        env_batch_size=ENV_BATCH_SIZE,
        train_steps_per_iter=TRAIN_STEPS_PER_ITER
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
