"""
Utility functions and visualization tools for DQN training.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(rewards, accuracies, losses, save_path='training_curves.png'):
    """
    Plot training curves: rewards, accuracies, and losses.

    Args:
        rewards: List of episode rewards
        accuracies: List of episode accuracies
        losses: List of episode losses
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot rewards
    axes[0].plot(rewards, alpha=0.3, label='Episode Reward')
    axes[0].plot(moving_average(rewards, window=50), label='Moving Average (50 eps)', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracies
    axes[1].plot(accuracies, alpha=0.3, label='Episode Accuracy')
    axes[1].plot(moving_average(accuracies, window=50), label='Moving Average (50 eps)', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Episode Accuracies')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Plot losses
    axes[2].plot(losses, alpha=0.3, label='Episode Loss')
    axes[2].plot(moving_average(losses, window=50), label='Moving Average (50 eps)', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def moving_average(data, window=50):
    """Calculate moving average of data"""
    if len(data) < window:
        return data
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')


def print_training_summary(rewards, accuracies, seq_length):
    """Print summary statistics of training"""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    total_episodes = len(rewards)
    final_100_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    final_100_accuracies = accuracies[-100:] if len(accuracies) >= 100 else accuracies

    print(f"Total Episodes: {total_episodes}")
    print(f"\nFinal 100 Episodes:")
    print(f"  Average Reward: {np.mean(final_100_rewards):.2f}/{seq_length}")
    print(f"  Average Accuracy: {np.mean(final_100_accuracies):.2%}")
    print(f"  Best Accuracy: {np.max(final_100_accuracies):.2%}")
    print(f"  Worst Accuracy: {np.min(final_100_accuracies):.2%}")
    print(f"  Std Dev: {np.std(final_100_accuracies):.4f}")

    print(f"\nOverall:")
    print(f"  Best Episode Accuracy: {np.max(accuracies):.2%} (Episode {np.argmax(accuracies) + 1})")
    print(f"  Final Episode Accuracy: {accuracies[-1]:.2%}")

    print("=" * 60)


def generate_sequence_with_model(model, seq_length, device, sep_token=2, temperature=1.0, greedy=True):
    """
    Generate a sequence using the trained model.

    Args:
        model: Trained transformer model
        seq_length: Length of sequence to generate
        device: Device (cpu or cuda)
        sep_token: Separator token ID
        temperature: Sampling temperature
        greedy: If True, use greedy decoding; otherwise sample

    Returns:
        Generated sequence (without sep token)
    """
    model.eval()

    with torch.no_grad():
        # Start with sep token
        sequence = [sep_token]

        for _ in range(seq_length):
            # Convert to tensor
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)

            # Get logits
            logits = model(input_tensor)  # (1, seq_len, vocab_size)

            # Get logits for last position and only tokens 0, 1
            last_logits = logits[0, -1, :2] / temperature

            if greedy:
                # Greedy: choose argmax
                next_token = torch.argmax(last_logits).item()
            else:
                # Sample from distribution
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            sequence.append(next_token)

    model.train()
    return sequence[1:]  # Remove sep token


def compare_models(model1, model2, env, num_episodes=10):
    """
    Compare two models on the same set of target sequences.

    Args:
        model1: First model
        model2: Second model
        env: Environment
        num_episodes: Number of episodes to compare

    Returns:
        Comparison statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    results = {
        'model1_accuracies': [],
        'model2_accuracies': [],
        'targets': [],
        'model1_outputs': [],
        'model2_outputs': []
    }

    for episode in range(num_episodes):
        obs, info = env.reset()
        target = info['target_sequence']

        # Generate with model 1
        seq1 = generate_sequence_with_model(model1, env.seq_length, device)

        # Generate with model 2
        seq2 = generate_sequence_with_model(model2, env.seq_length, device)

        # Calculate accuracies
        acc1 = np.mean(np.array(seq1) == target)
        acc2 = np.mean(np.array(seq2) == target)

        results['targets'].append(target)
        results['model1_outputs'].append(seq1)
        results['model2_outputs'].append(seq2)
        results['model1_accuracies'].append(acc1)
        results['model2_accuracies'].append(acc2)

    print(f"\nModel 1 Average Accuracy: {np.mean(results['model1_accuracies']):.2%}")
    print(f"Model 2 Average Accuracy: {np.mean(results['model2_accuracies']):.2%}")

    return results


if __name__ == "__main__":
    print("Testing utility functions...")

    # Test plotting with dummy data
    dummy_rewards = [np.random.rand() * 10 for _ in range(100)]
    dummy_accuracies = [min(1.0, 0.5 + i * 0.005 + np.random.rand() * 0.1) for i in range(100)]
    dummy_losses = [max(0, 1.0 - i * 0.01 + np.random.rand() * 0.2) for i in range(100)]

    plot_training_curves(dummy_rewards, dummy_accuracies, dummy_losses, 'test_curves.png')
    print_training_summary(dummy_rewards, dummy_accuracies, seq_length=10)

    print("\nUtility functions test passed!")
