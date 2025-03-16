import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

def create_probability_visualizations(results, output_dir="results"):
    """Create and save visualizations focusing on probability distributions"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'sans-serif'

    # 1. Overall comparison of models
    model_names = list(results.keys())
    metrics = ['accuracy', 'close_accuracy_1', 'close_accuracy_2', 'correlation']
    metric_titles = ['Exact Match Accuracy', 'Close Accuracy (±1)', 'Close Accuracy (±2)', 'Correlation with Gold']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        values = [results[model][metric] for model in model_names]
        axes[i].bar(model_names, values, color=sns.color_palette("muted", len(model_names)))
        axes[i].set_title(title)

        # Set y-axis limits appropriately
        if 'accuracy' in metric:
            axes[i].set_ylim(0, max(1.0, max(values) * 1.1))
        elif metric == 'correlation':
            axes[i].set_ylim(0, 1.0)

        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha='center')

        # Format y-axis as percentage for accuracy metrics
        if 'accuracy' in metric:
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300)
    plt.savefig(f"{output_dir}/model_comparison.pdf", dpi=300)
    plt.close()

    # 2. Probability Distribution vs. CommitmentBank Distribution
    for model_name, model_data in results.items():
        if 'z_score_bins' not in model_data:
            continue

        z_bins = np.array(model_data['z_score_bins']['bins'])
        ratings = np.array(model_data['z_score_bins']['ratings'])
        prob_mean = model_data['z_score_bins']['mean']
        prob_std = model_data['z_score_bins']['std_dev']
        gold_mean = model_data['z_score_bins']['gold_mean']
        gold_std = model_data['z_score_bins']['gold_std']

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Create a color map for different bins
        cmap = plt.cm.get_cmap('viridis', len(ratings))
        colors = [cmap(i) for i in range(len(ratings))]

        # Calculate rating frequencies
        rating_counts = {}
        for i, r in enumerate(ratings):
            if i < len(z_bins) - 1:
                rating_counts[r] = stats.norm.cdf(z_bins[i+1]) - stats.norm.cdf(z_bins[i])

        # Create two plots - one for the probability density and one for the distribution
        # First, plot the CommitmentBank distribution
        rating_x = np.array(list(rating_counts.keys()))
        rating_y = np.array(list(rating_counts.values()))
        ax.bar(rating_x, rating_y, alpha=0.5, color='gray', label='CommitmentBank')

        # Add a normal distribution curve to represent the z-score standardization
        x = np.linspace(-3, 3, 1000)
        y = stats.norm.pdf(x, loc=0, scale=1)
        # Scale to match the bar heights
        y = y * max(rating_y) / max(y)

        ax2 = ax.twinx()  # Create a second y-axis
        ax2.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')

        # Add vertical lines for the z-score boundaries
        for i, (boundary, rating) in enumerate(zip(z_bins[1:-1], ratings[:-1])):
            ax.axvline(x=rating + 0.5, color='k', linestyle='--', alpha=0.5)
            ax.text(rating + 0.5, ax.get_ylim()[1]*0.95, f"z={boundary:.2f}",
                   ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7))

        # Add labels and title
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")
        ax2.set_ylabel("Density")
        ax.set_title(f"{model_name}: Distribution Matching with CommitmentBank")

        # Set x-axis to show only integer ratings
        ax.set_xticks(rating_x)

        # Add legend for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Add annotations with distribution information
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        textstr = f"CommitmentBank: μ={gold_mean:.2f}, σ={gold_std:.2f}\nProbabilities: μ={prob_mean:.2f}, σ={prob_std:.2f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_distribution.png", dpi=300)
        plt.savefig(f"{output_dir}/{model_name}_distribution.pdf", dpi=300)
        plt.close()

    # 3. Confusion matrices with probability information
    for model_name, model_results in results.items():
        if 'confusion_matrix' not in model_results:
            continue

        cm = np.array(model_results['confusion_matrix'])

        # Create a normalized version (by row)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                cm_norm[i] = cm[i] / row_sums[i]

        # Create a custom colormap that goes from white to blue
        cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#ffffff', '#0000ff'])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)

        # Add labels
        tick_marks = np.arange(-3, 4)
        ax.set_xticks(np.arange(7))
        ax.set_yticks(np.arange(7))
        ax.set_xticklabels(tick_marks)
        ax.set_yticklabels(tick_marks)

        # Add text annotations
        thresh = cm_norm.max() / 2.
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                if cm_norm[i, j] > 0:
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                           ha="center", va="center",
                           color="white" if cm_norm[i, j] > thresh else "black",
                           fontsize=9)

        # Set labels and title
        ax.set_xlabel("Predicted Rating")
        ax.set_ylabel("Gold Rating")
        ax.set_title(f"{model_name}: Confusion Matrix")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized Frequency")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png", dpi=300)
        plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.pdf", dpi=300)
        plt.close()

    # 4. Probability vs. Gold Rating
    for model_name, pred_data in predictions.items():
        df = pd.DataFrame(pred_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create box plot of probabilities by gold rating
        sns.boxplot(x='gold', y='prob_true', data=df, ax=ax)

        # Add scatter plot with jitter for individual points
        sns.stripplot(x='gold', y='prob_true', data=df, ax=ax,
                     size=3, alpha=0.3, jitter=True, color='black')

        # Add reference line for ideal mapping
        gold_ratings = sorted(df['gold'].unique())
        ideal_probs = [(g + 3) / 6 for g in gold_ratings]  # Map from [-3,3] to [0,1]
        ax.plot(gold_ratings, ideal_probs, 'r--', alpha=0.7, label='Ideal Linear Mapping')

        # Set labels and title
        ax.set_xlabel("Gold Rating (-3 to +3)")
        ax.set_ylabel("Probability of TRUE")
        ax.set_title(f"{model_name}: TRUE Probability vs. Gold Rating")

        # Set y-axis limits
        ax.set_ylim(-0.05, 1.05)

        # Add legend
        ax.legend()

        # Add correlation coefficient as text
        corr = results[model_name]["correlation"]
        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_prob_vs_gold.png", dpi=300)
        plt.savefig(f"{output_dir}/{model_name}_prob_vs_gold.pdf", dpi=300)
        plt.close()

    return True

def create_performance_comparison_figures(results, output_dir="results"):
    """Create visualizations comparing model performance across different metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall accuracy metrics
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Extract model names and metrics
    model_names = list(results.keys())
    close_accuracy_1 = [results[model]['close_accuracy_1'] for model in model_names]
    close_accuracy_2 = [results[model]['close_accuracy_2'] for model in model_names]
    exact_accuracy = [results[model]['accuracy'] for model in model_names]
    
    # Set width and positions for bars
    x = np.arange(len(model_names))
    width = 0.25
    
    # Create grouped bar chart
    ax.bar(x - width, exact_accuracy, width, label='Exact Match', color='#1f77b4')
    ax.bar(x, close_accuracy_1, width, label='Within ±1', color='#ff7f0e')
    ax.bar(x + width, close_accuracy_2, width, label='Within ±2', color='#2ca02c')
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_accuracy_comparison.png", dpi=300)
    plt.savefig(f"{output_dir}/model_accuracy_comparison.pdf", dpi=300)
    plt.close()
    
    # 2. Error metrics
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Extract error metrics
    mae = [results[model]['mae'] for model in model_names]
    rmse = [results[model]['rmse'] for model in model_names]
    
    # Set width and positions for bars
    x = np.arange(len(model_names))
    width = 0.35
    
    # Create grouped bar chart
    ax.bar(x - width/2, mae, width, label='MAE', color='#d62728')
    ax.bar(x + width/2, rmse, width, label='RMSE', color='#9467bd')
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Error')
    ax.set_title('Model Error Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_error_metrics.png", dpi=300)
    plt.savefig(f"{output_dir}/model_error_metrics.pdf", dpi=300)
    plt.close()
    
    # 3. Performance by embedding type
    embedding_types = ['negation', 'question', 'modal', 'conditional']
    models_to_plot = [m for m in model_names if 'embedding_metrics' in results[m]]
    
    if models_to_plot:
        plt.figure(figsize=(14, 7))
        ax = plt.gca()
        
        # Set up positions for grouped bars
        n_groups = len(embedding_types)
        n_models = len(models_to_plot)
        width = 0.8 / n_models
        
        # Create grouped bar chart for each embedding type
        for i, model in enumerate(models_to_plot):
            embeddings = results[model].get('embedding_metrics', {})
            values = [embeddings.get(embed, {}).get('accuracy', 0) for embed in embedding_types]
            offsets = np.arange(n_groups) - ((n_models-1)/2 - i) * width
            
            ax.bar(offsets, values, width, label=model)
        
        # Add labels and title
        ax.set_xlabel('Embedding Type')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy by Embedding Environment', fontweight='bold')
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels([e.capitalize() for e in embedding_types])
        ax.legend()
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_by_embedding.png", dpi=300)
        plt.savefig(f"{output_dir}/accuracy_by_embedding.pdf", dpi=300)
        plt.close()
    
    # 4. Performance by predicate verb
    common_verbs = []
    for model in model_names:
        if 'verb_metrics' in results[model]:
            verbs = list(results[model]['verb_metrics'].keys())
            if not common_verbs:
                common_verbs = verbs
            else:
                common_verbs = [v for v in common_verbs if v in verbs]
    
    if common_verbs:
        # Limit to top 8 most common verbs
        verb_counts = {}
        for verb in common_verbs:
            verb_counts[verb] = results[model_names[0]]['verb_metrics'][verb]['count']
        
        top_verbs = sorted(verb_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        top_verb_names = [v[0] for v in top_verbs]
        
        plt.figure(figsize=(14, 7))
        ax = plt.gca()
        
        # Set up positions for grouped bars
        n_groups = len(top_verb_names)
        n_models = len(model_names)
        width = 0.8 / n_models
        
        # Create grouped bar chart for each verb
        for i, model in enumerate(model_names):
            verbs = results[model].get('verb_metrics', {})
            values = [verbs.get(verb, {}).get('accuracy', 0) for verb in top_verb_names]
            offsets = np.arange(n_groups) - ((n_models-1)/2 - i) * width
            
            ax.bar(offsets, values, width, label=model)
        
        # Add labels and title
        ax.set_xlabel('Predicate Verb')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy by Predicate Verb', fontweight='bold')
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels(top_verb_names)
        ax.legend()
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_by_predicate.png", dpi=300)
        plt.savefig(f"{output_dir}/accuracy_by_predicate.pdf", dpi=300)
        plt.close()

def create_error_distribution_figure(results, predictions, output_dir="results"):
    """Create a visualization of error distributions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select models to visualize
    model_names = list(results.keys())
    
    # 1. Error magnitude distribution
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    for model_name in model_names:
        if model_name in predictions:
            pred_df = pd.DataFrame(predictions[model_name])
            # Calculate error (absolute)
            error_dist = pred_df['error'].value_counts().sort_index()
            
            # Calculate percentage distribution
            error_pct = error_dist / len(pred_df) * 100
            
            # Plot as line
            ax.plot(error_pct.index, error_pct.values, 'o-', label=model_name, linewidth=2, markersize=8)
    
    # Add labels and title
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Percentage of Predictions (%)')
    ax.set_title('Distribution of Error Magnitudes', fontweight='bold')
    ax.set_xticks(range(7))  # 0 to 6 errors
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_magnitude_distribution.png", dpi=300)
    plt.savefig(f"{output_dir}/error_magnitude_distribution.pdf", dpi=300)
    plt.close()
    
    # 2. Error by gold rating
    # Select one model for this visualization
    selected_model = next(iter(predictions))
    if selected_model in predictions:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        pred_df = pd.DataFrame(predictions[selected_model])
        
        # Create box plot
        sns.boxplot(x='gold', y='error', data=pred_df, ax=ax, palette='Blues')
        
        # Add jittered points for individual data points
        sns.stripplot(x='gold', y='error', data=pred_df, ax=ax, 
                     size=4, alpha=0.3, jitter=True, color='black')
        
        # Add labels and title
        ax.set_xlabel('Gold Rating')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'Error by Gold Rating: {selected_model}', fontweight='bold')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_by_gold_rating.png", dpi=300)
        plt.savefig(f"{output_dir}/error_by_gold_rating.pdf", dpi=300)
        plt.close()

def create_factive_analysis(results, output_dir="results"):
    """Create a visualization comparing factive vs. non-factive predicates"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define known factive and non-factive predicates
    factive_predicates = {'know', 'realize', 'find', 'discover', 'notice', 'remember', 'forget'}
    nonfactive_predicates = {'believe', 'think', 'say', 'claim', 'suggest', 'hope', 'want'}
    
    # Collect data on factive vs. non-factive performance
    factive_data = []
    
    for model_name, model_results in results.items():
        if 'verb_metrics' in model_results:
            verb_metrics = model_results['verb_metrics']
            
            # Calculate average metrics for factive verbs
            factive_verbs = {v: metrics for v, metrics in verb_metrics.items() 
                           if v.lower() in factive_predicates}
            nonfactive_verbs = {v: metrics for v, metrics in verb_metrics.items() 
                              if v.lower() in nonfactive_predicates}
            
            if factive_verbs:
                factive_acc = np.mean([m['accuracy'] for m in factive_verbs.values()])
                factive_close1 = np.mean([m['close_accuracy_1'] for m in factive_verbs.values()])
                factive_count = sum(m['count'] for m in factive_verbs.values())
                
                factive_data.append({
                    'Model': model_name,
                    'Type': 'Factive',
                    'Accuracy': factive_acc,
                    'Close_Accuracy_1': factive_close1,
                    'Count': factive_count
                })
            
            if nonfactive_verbs:
                nonfactive_acc = np.mean([m['accuracy'] for m in nonfactive_verbs.values()])
                nonfactive_close1 = np.mean([m['close_accuracy_1'] for m in nonfactive_verbs.values()])
                nonfactive_count = sum(m['count'] for m in nonfactive_verbs.values())
                
                factive_data.append({
                    'Model': model_name,
                    'Type': 'Non-factive',
                    'Accuracy': nonfactive_acc,
                    'Close_Accuracy_1': nonfactive_close1,
                    'Count': nonfactive_count
                })
    
    if factive_data:
        # Convert to DataFrame
        df = pd.DataFrame(factive_data)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Set up positions for grouped bars
        models = df['Model'].unique()
        n_models = len(models)
        width = 0.35
        
        # Create grouped bar chart for factive vs. non-factive
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            
            # Get factive and non-factive values
            factive_value = model_data[model_data['Type'] == 'Factive']['Accuracy'].values[0] if 'Factive' in model_data['Type'].values else 0
            nonfactive_value = model_data[model_data['Type'] == 'Non-factive']['Accuracy'].values[0] if 'Non-factive' in model_data['Type'].values else 0
            
            # Plot bars
            ax.bar([i - width/2], [factive_value], width, label=f'Factive ({model})' if i == 0 else "_nolegend_", color='#1f77b4')
            ax.bar([i + width/2], [nonfactive_value], width, label=f'Non-factive ({model})' if i == 0 else "_nolegend_", color='#ff7f0e')
            
            # Add labels
            ax.text(i - width/2, factive_value + 0.01, f"{factive_value:.3f}", ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, nonfactive_value + 0.01, f"{nonfactive_value:.3f}", ha='center', va='bottom', fontsize=9)
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Factive vs. Non-factive Predicate Accuracy', fontweight='bold')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.legend()
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/factive_vs_nonfactive.png", dpi=300)
        plt.savefig(f"{output_dir}/factive_vs_nonfactive.pdf", dpi=300)
        plt.close()

def create_summary_figure(results, output_dir="results"):
    """Create a publication-quality summary figure with key results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Overall performance comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    model_names = list(results.keys())
    close_accuracy_1 = [results[model]['close_accuracy_1'] for model in model_names]
    
    # Create bar chart
    bars = ax1.bar(model_names, close_accuracy_1, color=sns.color_palette("Blues", len(model_names)))
    
    # Add value labels
    for bar, value in zip(bars, close_accuracy_1):
        ax1.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.3f}", 
                ha='center', va='bottom', fontsize=9)
    
    # Add labels and title
    ax1.set_ylabel('Close Accuracy (±1)')
    ax1.set_title('Model Performance (Close ±1)', fontweight='bold')
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Error metrics comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    mae_values = [results[model]['mae'] for model in model_names]
    
    # Create bar chart
    bars = ax2.bar(model_names, mae_values, color=sns.color_palette("Reds", len(model_names)))
    
    # Add value labels
    for bar, value in zip(bars, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 0.05, f"{value:.3f}", 
                ha='center', va='bottom', fontsize=9)
    
    # Add labels and title
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.set_title('Model Error (MAE)', fontweight='bold')
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Confusion matrix for best model (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Find model with highest close_accuracy_1
    best_model = model_names[np.argmax(close_accuracy_1)]
    
    if 'confusion_matrix' in results[best_model]:
        cm = np.array(results[best_model]['confusion_matrix'])
        
        # Create a normalized version (by row)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                cm_norm[i] = cm[i] / row_sums[i]
        
        # Create confusion matrix heatmap
        cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#ffffff', '#0000ff'])
        im = ax3.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        
        # Add labels
        tick_marks = np.arange(-3, 4)
        ax3.set_xticks(np.arange(7))
        ax3.set_yticks(np.arange(7))
        ax3.set_xticklabels(tick_marks)
        ax3.set_yticklabels(tick_marks)
        
        # Add text annotations (simplified for clarity)
        thresh = cm_norm.max() / 2.
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                if cm_norm[i, j] > 0.2:  # Only show major values
                    ax3.text(j, i, f"{cm_norm[i, j]:.2f}",
                           ha="center", va="center",
                           color="white" if cm_norm[i, j] > thresh else "black")
        
        # Set labels and title
        ax3.set_xlabel("Predicted Rating")
        ax3.set_ylabel("Gold Rating")
        ax3.set_title(f"Confusion Matrix: {best_model}", fontweight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label("Normalized Frequency")
    
    # 4. Performance by embedding type (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Select models with embedding metrics
    models_with_embedding = [m for m in model_names if 'embedding_metrics' in results[m]]
    
    if models_with_embedding:
        embedding_types = ['negation', 'question', 'modal', 'conditional']
        
        # Set up positions for grouped bars
        n_groups = len(embedding_types)
        n_models = len(models_with_embedding)
        width = 0.8 / n_models
        
        # Create grouped bar chart for each embedding type
        for i, model in enumerate(models_with_embedding):
            embeddings = results[model].get('embedding_metrics', {})
            values = [embeddings.get(embed, {}).get('close_accuracy_1', 0) for embed in embedding_types]
            offsets = np.arange(n_groups) - ((n_models-1)/2 - i) * width
            
            ax4.bar(offsets, values, width, label=model)
        
        # Add labels and title
        ax4.set_xlabel('Embedding Type')
        ax4.set_ylabel('Close Accuracy (±1)')
        ax4.set_title('Performance by Embedding Type', fontweight='bold')
        ax4.set_xticks(np.arange(n_groups))
        ax4.set_xticklabels([e.capitalize() for e in embedding_types])
        ax4.legend(fontsize=8)
        
        # Format y-axis as percentage
        ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Add grid
        ax4.grid(True, linestyle='--', alpha=0.5)
    
    # Add overall title
    fig.suptitle('Language Models have Commitment Issues: Key Results', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the title
    plt.savefig(f"{output_dir}/summary_figure.png", dpi=300)
    plt.savefig(f"{output_dir}/summary_figure.pdf", dpi=300)
    plt.close()

def make_camera_ready_figures(results_file="commitmentbank_distribution_results.json", prediction_dir=".", output_dir="results"):
    """Generate all figures for the paper"""
    # Load results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded results from {results_file}")
    except FileNotFoundError:
        print(f"Results file {results_file} not found.")
        return
    
    # Load prediction files
    predictions = {}
    for model in results.keys():
        pred_file = f"{prediction_dir}/{model}_predictions.csv"
        try:
            predictions[model] = pd.read_csv(pred_file)
            print(f"Loaded predictions for {model}")
        except FileNotFoundError:
            print(f"Prediction file for {model} not found.")
    
    # Create all figures
    create_probability_visualizations(results, output_dir)
    create_performance_comparison_figures(results, output_dir)
    create_error_distribution_figure(results, predictions, output_dir)
    create_factive_analysis(results, output_dir)
    create_summary_figure(results, output_dir)
    
    print(f"All figures saved to {output_dir}/")

if __name__ == "__main__":
    # Generate all figures
    make_camera_ready_figures()