import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class SequenceGenerationEnv(gym.Env):
    """
    Environment for training a model to generate binary sequences.

    The agent must generate a sequence of 0s and 1s token by token.
    Reward: +1 if the generated token at position k matches the target at position k+1
    (considering the prefix from position 0 to k)
    """

    metadata = {'render_modes': []}

    def __init__(self, seq_length=10, vocab_size=3, sep_token=2):
        """
        Args:
            seq_length: Length of the target binary sequence (n)
            vocab_size: Size of vocabulary (default 3: 0, 1, sep)
            sep_token: Token ID for separator (default 2)
        """
        super(SequenceGenerationEnv, self).__init__()

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.sep_token = sep_token

        # Action space: choose token 0 or 1
        self.action_space = spaces.Discrete(2)

        # Observation space: [target_sequence, sep_token, generated_sequence]
        # Max length is seq_length (target) + 1 (sep) + seq_length (generated)
        self.observation_space = spaces.Box(
            low=0,
            high=vocab_size-1,
            shape=(2 * seq_length + 1,),
            dtype=np.int32
        )

        # State variables
        self.target_sequence = None
        self.current_sequence = None
        self.current_position = None
        self.max_steps = seq_length

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation"""
        super().reset(seed=seed)

        # Generate random target sequence of 0s and 1s
        if seed is not None:
            np.random.seed(seed)
        self.target_sequence = np.random.randint(0, 2, size=self.seq_length)
        # self.target_sequence = np.zeros(self.seq_length, dtype=int)

        # Initialize current sequence with sep token
        self.current_sequence = [self.sep_token]
        self.current_position = 0  # Position in target sequence

        # Create observation (padded to fixed length)
        observation = self._get_observation()
        info = {
            'target_sequence': self.target_sequence.copy(),
            'current_position': self.current_position
        }

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: 0 or 1 (the token to generate)

        Returns:
            observation: current sequence state
            reward: 1 if action matches target[current_position], else 0
            terminated: True if sequence is complete
            truncated: False (not used here)
            info: additional information
        """
        # Check if action is valid
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Check if episode is already done
        if self.current_position >= self.seq_length:
            raise RuntimeError("Episode is already done. Call reset() first.")

        # Get the target token at current position
        target_token = self.target_sequence[self.current_position]

        # Append action to current sequence
        self.current_sequence.append(action)
        self.current_position += 1

        # Calculate reward: +1 only if ALL tokens so far match target, else 0
        # Check if all generated tokens match the target prefix
        generated_tokens = self.current_sequence[1:]  # Exclude sep token
        target_prefix = self.target_sequence[:len(generated_tokens)]
        all_match = np.array_equal(generated_tokens, target_prefix)
        reward = 1.0 if all_match else 0.0

        # Check if episode is done
        terminated = (self.current_position >= self.seq_length) or not all_match
        truncated = False

        # Get new observation
        observation = self._get_observation()

        info = {
            'target_sequence': self.target_sequence.copy(),
            'current_position': self.current_position,
            'generated_sequence': np.array(self.current_sequence[1:]),  # Exclude sep token
            'target_token': target_token,
            'predicted_token': action,
            'match': action == target_token
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Get current observation: [target_sequence, sep_token, generated_sequence].

        Format: [t0, t1, ..., tn-1, sep, g0, g1, ..., gk]
        where ti is target token, sep is separator, gi is generated token

        Returns:
            numpy array of shape (seq_length + 1 + len(generated),)
        """
        # Get generated sequence (excluding the initial sep token from current_sequence)
        generated = self.current_sequence[1:]  # Skip the sep token

        # Build observation: target + sep + generated (no padding)
        obs = np.concatenate([
            self.target_sequence,
            [self.sep_token],
            generated
        ]).astype(np.int32)

        return obs

    def get_current_sequence_tensor(self):
        """
        Get current sequence as PyTorch tensor for model input.

        Returns:
            tensor of shape (1, current_length)
        """
        return torch.tensor([self.current_sequence], dtype=torch.long)

    def render(self):
        """Render the current state"""
        if self.current_sequence is None:
            return

        print(f"Target:    {self.target_sequence}")
        print(f"Generated: {self.current_sequence[1:]}")  # Exclude sep token
        print(f"Position:  {self.current_position}/{self.seq_length}")


if __name__ == "__main__":
    # Test the environment
    print("Testing SequenceGenerationEnv...")

    env = SequenceGenerationEnv(seq_length=5)

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Target sequence: {info['target_sequence']}")

    # Test a few steps
    print("\n--- Running episode ---")
    total_reward = 0
    terminated = False
    step_count = 0

    while not terminated:
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        print(f"Step {step_count}: action={action}, target={info['target_token']}, "
              f"reward={reward}, match={info['match']}")

    print(f"\n--- Episode finished ---")
    print(f"Total reward: {total_reward}/{env.seq_length}")
    print(f"Accuracy: {total_reward/env.seq_length:.2%}")
    env.render()

    # Test multiple episodes
    print("\n--- Testing multiple episodes ---")
    num_episodes = 3
    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        terminated = False

        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {ep+1}: Reward = {total_reward}/{env.seq_length}")

    print("\nEnvironment test passed!")
