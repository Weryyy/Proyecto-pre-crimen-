"""
Simple Example: Risk Seed Evolution Demonstration

This script demonstrates how the Beta distribution is used to generate
and evolve risk seeds over time.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)


def generate_risk_seed(alpha=2.0, beta=5.0):
    """Generate initial risk seed using Beta distribution"""
    return np.random.beta(alpha, beta)


def evolve_risk_seed(current_risk, interactions, location_crime_rate, time_delta=1.0):
    """Evolve risk seed based on environmental factors"""
    # Adjust Beta distribution parameters based on current state
    alpha_new = 2.0 + current_risk * 3.0 + interactions * 0.1
    beta_new = 5.0 - location_crime_rate * 2.0
    beta_new = max(beta_new, 1.0)  # Ensure positive
    
    # Generate new sample
    new_risk_sample = np.random.beta(alpha_new, beta_new)
    
    # Combine with momentum
    evolved_risk = 0.7 * current_risk + 0.3 * new_risk_sample
    
    # Add environmental influence
    env_factor = location_crime_rate * 0.1 * time_delta
    evolved_risk = min(1.0, evolved_risk + env_factor)
    
    return evolved_risk


def simulate_population(num_people=100, num_timesteps=50):
    """Simulate risk evolution for a population"""
    # Initialize population
    risks = np.array([generate_risk_seed() for _ in range(num_people)])
    risk_history = [risks.copy()]
    
    # Simulate evolution
    for t in range(num_timesteps):
        new_risks = []
        for risk in risks:
            # Simulate random interactions and locations
            interactions = np.random.randint(0, 10)
            crime_rate = np.random.beta(2, 8)  # Low crime areas more common
            
            new_risk = evolve_risk_seed(risk, interactions, crime_rate)
            new_risks.append(new_risk)
        
        risks = np.array(new_risks)
        risk_history.append(risks.copy())
    
    return np.array(risk_history)


def plot_risk_evolution(risk_history, output_dir='examples'):
    """Plot risk evolution over time"""
    Path(output_dir).mkdir(exist_ok=True)
    
    num_timesteps, num_people = risk_history.shape
    
    # Plot 1: Individual trajectories
    plt.figure(figsize=(12, 6))
    
    # Plot sample trajectories
    sample_indices = np.random.choice(num_people, size=10, replace=False)
    for idx in sample_indices:
        plt.plot(risk_history[:, idx], alpha=0.6, linewidth=1)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Risk Score')
    plt.title('Risk Evolution Over Time (Sample Individuals)')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.savefig(f'{output_dir}/risk_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/risk_trajectories.png")
    plt.close()
    
    # Plot 2: Distribution evolution
    plt.figure(figsize=(12, 6))
    
    timesteps_to_plot = [0, 10, 25, 49]
    for t in timesteps_to_plot:
        plt.hist(risk_history[t], bins=30, alpha=0.5, label=f'Time {t}', density=True)
    
    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.title('Risk Score Distribution at Different Time Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/risk_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/risk_distribution.png")
    plt.close()
    
    # Plot 3: Statistics over time
    plt.figure(figsize=(12, 6))
    
    mean_risk = risk_history.mean(axis=1)
    std_risk = risk_history.std(axis=1)
    high_risk_count = (risk_history > 0.5).sum(axis=1)
    
    plt.subplot(1, 2, 1)
    plt.plot(mean_risk, label='Mean Risk', linewidth=2)
    plt.fill_between(range(num_timesteps), 
                     mean_risk - std_risk, 
                     mean_risk + std_risk, 
                     alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Time Steps')
    plt.ylabel('Risk Score')
    plt.title('Population Risk Statistics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.subplot(1, 2, 2)
    plt.plot(high_risk_count, color='red', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Count')
    plt.title('High-Risk Individuals (Risk > 0.5)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/risk_statistics.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/risk_statistics.png")
    plt.close()


def demonstrate_beta_distribution(output_dir='examples'):
    """Demonstrate Beta distribution properties"""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Different Beta distribution parameters
    params = [
        (2, 5, 'Low Risk (α=2, β=5)'),
        (5, 2, 'High Risk (α=5, β=2)'),
        (2, 2, 'Uniform-ish (α=2, β=2)'),
        (5, 5, 'Moderate (α=5, β=5)')
    ]
    
    x = np.linspace(0, 1, 1000)
    
    for i, (alpha, beta, label) in enumerate(params, 1):
        plt.subplot(2, 2, i)
        
        # Generate samples
        samples = np.random.beta(alpha, beta, 10000)
        
        # Plot histogram
        plt.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
        
        # Plot theoretical PDF
        from scipy.stats import beta as beta_dist
        pdf = beta_dist.pdf(x, alpha, beta)
        plt.plot(x, pdf, 'r-', linewidth=2, label='PDF')
        
        plt.xlabel('Risk Value')
        plt.ylabel('Density')
        plt.title(f'{label}\nMean={samples.mean():.3f}, Std={samples.std():.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/beta_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/beta_distributions.png")
    plt.close()


def main():
    """Run all demonstrations"""
    print("=" * 60)
    print("Risk Seed Evolution Demonstration")
    print("=" * 60)
    
    # Create output directory
    output_dir = 'examples'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Demonstrate Beta distributions
    print("\n1. Demonstrating Beta Distribution Properties...")
    demonstrate_beta_distribution(output_dir)
    
    # Simulate population
    print("\n2. Simulating Population Risk Evolution...")
    print("   - Population size: 100 individuals")
    print("   - Time steps: 50")
    
    risk_history = simulate_population(num_people=100, num_timesteps=50)
    
    print(f"\n   Initial statistics:")
    print(f"   - Mean risk: {risk_history[0].mean():.3f}")
    print(f"   - Std risk: {risk_history[0].std():.3f}")
    print(f"   - High-risk count: {(risk_history[0] > 0.5).sum()}")
    
    print(f"\n   Final statistics:")
    print(f"   - Mean risk: {risk_history[-1].mean():.3f}")
    print(f"   - Std risk: {risk_history[-1].std():.3f}")
    print(f"   - High-risk count: {(risk_history[-1] > 0.5).sum()}")
    
    # Plot results
    print("\n3. Generating Plots...")
    plot_risk_evolution(risk_history, output_dir)
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print(f"Results saved in '{output_dir}/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
