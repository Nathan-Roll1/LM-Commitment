import pandas as pd
import numpy as np
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.functional import softmax
from scipy import stats

# Load the CSV with comma delimiter
def load_commitmentbank_data(file_path="CommitmentBank-All.csv"):
    df = pd.read_csv(file_path, sep=",")
    print(f"Data loaded: {df.shape}")

    # Clean the dataset by removing rows with NaN in important columns
    df_clean = df.dropna(subset=['Context', 'Target', 'mean.noTarget'])
    print(f"Clean dataset: {df_clean.shape} (removed {len(df) - len(df_clean)} rows with missing values)")

    # Calculate the standard deviation of the gold ratings (mean.noTarget)
    global gold_ratings, gold_std, gold_mean, unique_ratings, rating_freqs, z_boundaries
    
    gold_ratings = df_clean['mean.noTarget'].apply(lambda x: round(float(x))).values
    gold_std = np.std(gold_ratings)
    gold_mean = np.mean(gold_ratings)
    print(f"CommitmentBank ratings: Mean={gold_mean:.3f}, StdDev={gold_std:.3f}")

    # Calculate the frequency distribution of the gold ratings
    unique_ratings, rating_counts = np.unique(gold_ratings, return_counts=True)
    rating_freqs = rating_counts / len(gold_ratings)

    # Calculate cumulative frequencies to find bin boundaries
    cum_freqs = np.cumsum(rating_freqs)
    cum_freqs = np.insert(cum_freqs, 0, 0)  # Insert 0 at the beginning

    # Convert to z-scores using the inverse CDF of the standard normal
    z_boundaries = stats.norm.ppf(cum_freqs)
    z_boundaries[0] = -float('inf')  # Replace -inf with a large negative number
    z_boundaries[-1] = float('inf')  # Replace inf with a large positive number

    print("\nCommitmentBank rating distribution:")
    print(f"Ratings: {unique_ratings.tolist()}")
    print(f"Frequencies: {rating_freqs.tolist()}")
    print(f"Z-score boundaries: {z_boundaries.tolist()}")
    
    return df_clean

def get_true_false_probabilities(row, model, tokenizer, device):
    """
    Evaluate the probability of true vs false for a given sample using a pretrained LLM.
    Optimized for non-instruction tuned models to naturally produce true/false continuations.

    Returns:
        - probability_true: The combined probability of "true" tokens
    """
    # Access the Context and Target columns
    context = row['Context']
    target = row['Target']
    prompt = row['Prompt']  # The clause content we're evaluating

    # Construct a more natural prompt for pretrained models that will lead to true/false continuation
    natural_prompt = f"""{context}
Speaker: {target}

(True/False) The claim that "{prompt}" is """

    # Tokenize with attention mask
    inputs = tokenizer(natural_prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get model outputs (logits)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Get logits for the next token
        probs = softmax(logits, dim=-1)[0]  # Convert to probabilities

    # Get token IDs for various forms of "true" and "false"
    true_tokens = []
    false_tokens = []

    # Various case and space variations
    for true_text in ["true", "True", "TRUE", " true", " True", " TRUE"]:
        true_token_ids = tokenizer.encode(true_text, add_special_tokens=False)
        true_tokens.extend(true_token_ids)

    for false_text in ["false", "False", "FALSE", " false", " False", " FALSE"]:
        false_token_ids = tokenizer.encode(false_text, add_special_tokens=False)
        false_tokens.extend(false_token_ids)

    # Some models might also use period-terminated tokens
    for true_text in ["true.", "True.", " true.", " True."]:
        true_token_ids = tokenizer.encode(true_text, add_special_tokens=False)
        if len(true_token_ids) == 1:  # Only add if it's a single token
            true_tokens.extend(true_token_ids)

    for false_text in ["false.", "False.", " false.", " False."]:
        false_token_ids = tokenizer.encode(false_text, add_special_tokens=False)
        if len(false_token_ids) == 1:  # Only add if it's a single token
            false_tokens.extend(false_token_ids)

    # Remove duplicates
    true_tokens = list(set(true_tokens))
    false_tokens = list(set(false_tokens))

    # Debug info to verify tokenization (only for the first few examples)
    if row.get('idx', 0) < 5:  # Show for first 5 examples
        print("\nTokenization for true/false:")
        for token_id in true_tokens:
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            print(f"True token: {token} (ID: {token_id})")
        for token_id in false_tokens:
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            print(f"False token: {token} (ID: {token_id})")

        # Print top predicted tokens for debugging
        top_probs, top_indices = torch.topk(probs, 10)
        print("\nTop 10 predicted tokens:")
        for p, idx in zip(top_probs, top_indices):
            token = tokenizer.convert_ids_to_tokens([idx])[0]
            print(f"{token}: {p.item():.4f}")

    # Calculate probabilities
    prob_true = sum(probs[token_id].item() for token_id in true_tokens)
    prob_false = sum(probs[token_id].item() for token_id in false_tokens)

    # Normalize to get a proportion between true and false (when both have non-zero probabilities)
    total_prob = prob_true + prob_false
    if total_prob > 0:
        norm_prob_true = prob_true / total_prob
    else:
        # If neither true nor false has probability, default to 0.5
        norm_prob_true = 0.5

    return norm_prob_true

def evaluate_model(model_name, df, save_predictions=False, prediction_file=None, max_samples=None):
    """
    Loads a model and tokenizer, then evaluates the model on CommitmentBank.
    Uses z-score bucketing for probability to rating conversion.

    Parameters:
    - model_name: Name/path of the model to evaluate
    - df: DataFrame containing CommitmentBank data
    - save_predictions: Whether to save the predictions to a file
    - prediction_file: File to save predictions to (if save_predictions is True)
    - max_samples: Maximum number of samples to evaluate (None for all)

    Returns: Dictionary of evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model {model_name} on {device}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Fix for tokenizers without a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Check if the model has a specific pad token ID and set it if needed
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # Show what token IDs correspond to "true" and "false"
    print("\nToken IDs for true/false:")
    for text in ["true", "True", "TRUE", " true", " True", " TRUE", "false", "False", "FALSE", " false", " False", " FALSE"]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(f"  {text:<8}: {token_ids} ({tokens})")

    # Prepare for evaluation
    all_data = []
    original_golds = []
    probability_trues = []

    # Limit samples if specified
    if max_samples is not None:
        df_eval = df.sample(n=min(max_samples, len(df)), random_state=42)
    else:
        df_eval = df

    # Process each item - first pass to get all probabilities
    print("Getting all probabilities...")
    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        # Add index to row for debugging purposes
        row_with_idx = row.copy()
        row_with_idx['idx'] = idx

        # Get model prediction (true probability)
        prob_true = get_true_false_probabilities(row_with_idx, model, tokenizer, device)

        # Get gold rating from the 'mean.noTarget' column and round it
        gold_original = int(round(float(row["mean.noTarget"])))

        # Store results
        probability_trues.append(prob_true)
        original_golds.append(gold_original)

        # Store data (will add predictions after z-score calculation)
        all_data.append({
            'idx': idx,
            'uID': row.get('uID', ''),
            'Verb': row.get('Verb', ''),
            'Embedding': row.get('Embedding', ''),
            'Context': row.get('Context', ''),
            'Target': row.get('Target', ''),
            'Prompt': row.get('Prompt', ''),
            'gold': gold_original,
            'prob_true': prob_true
        })

    # Convert probabilities to numpy array for z-score calculation
    probability_trues = np.array(probability_trues)

    # Calculate z-scores for the probabilities
    prob_mean = np.mean(probability_trues)
    prob_std = np.std(probability_trues)

    print(f"\nProbability distribution: Mean={prob_mean:.3f}, StdDev={prob_std:.3f}")

    if prob_std == 0:  # Handle edge case of zero standard deviation
        print("Warning: Zero standard deviation in probabilities. Using fixed mapping.")
        z_scores = np.zeros_like(probability_trues)
    else:
        # Calculate z-scores (standardized scores)
        z_scores = (probability_trues - prob_mean) / prob_std

    # Use the CommitmentBank rating distribution boundaries
    bins = z_boundaries
    ratings = unique_ratings

    # For each z-score, find which bin it falls into
    predictions = np.zeros_like(z_scores, dtype=int)
    for i, z in enumerate(z_scores):
        for bin_idx in range(len(bins) - 1):
            if bins[bin_idx] <= z < bins[bin_idx + 1]:
                predictions[i] = ratings[bin_idx]
                break

    # Add predictions to the data
    for i, data_point in enumerate(all_data):
        data_point['pred'] = int(predictions[i])
        data_point['z_score'] = float(z_scores[i])
        data_point['error'] = abs(predictions[i] - data_point['gold'])

    # Save detailed predictions if requested
    if save_predictions and prediction_file is not None:
        prediction_df = pd.DataFrame(all_data)
        prediction_df.to_csv(prediction_file, index=False)
        print(f"Saved detailed predictions to {prediction_file}")

    # Convert to numpy arrays for metrics calculation
    original_predictions = np.array(predictions)
    original_golds = np.array(original_golds)

    # Calculate detailed evaluation metrics
    metrics = calculate_metrics(original_predictions, original_golds, probability_trues, model_name)

    # Calculate per-predicate metrics
    if 'Verb' in df_eval.columns:
        metrics['verb_metrics'] = calculate_predicate_metrics(all_data)

    # Calculate per-embedding metrics
    if 'Embedding' in df_eval.columns:
        metrics['embedding_metrics'] = calculate_embedding_metrics(all_data)

    # Add z-score binning information to metrics
    metrics['z_score_bins'] = {
        'bins': bins.tolist(),
        'ratings': ratings.tolist(),
        'mean': float(prob_mean),
        'std_dev': float(prob_std),
        'gold_mean': float(gold_mean),
        'gold_std': float(gold_std)
    }

    return metrics, all_data

def bootstrap_ci(data, statistic, n_bootstraps=1000, ci=0.95):
    """
    Calculate bootstrap confidence intervals for a statistic.

    Parameters:
    - data: The data to bootstrap
    - statistic: Function that calculates the statistic of interest
    - n_bootstraps: Number of bootstrap samples
    - ci: Confidence interval level (default: 0.95 for 95% CI)

    Returns:
    - low: Lower bound of the confidence interval
    - high: Upper bound of the confidence interval
    """
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstraps):
        # Generate bootstrap sample (with replacement)
        indices = np.random.randint(0, n, size=n)
        sample = data[indices]

        # Calculate statistic on the bootstrap sample
        stat = statistic(sample)
        bootstrap_stats.append(stat)

    # Calculate confidence interval bounds
    alpha = (1 - ci) / 2
    low = np.percentile(bootstrap_stats, 100 * alpha)
    high = np.percentile(bootstrap_stats, 100 * (1 - alpha))

    return low, high

def calculate_metrics(predictions, golds, probability_trues, model_name):
    """Calculate and print comprehensive evaluation metrics with confidence intervals"""
    # Basic accuracy metrics
    accuracy = np.mean(predictions == golds)
    close_accuracy_1 = np.mean(np.abs(predictions - golds) <= 1)
    close_accuracy_2 = np.mean(np.abs(predictions - golds) <= 2)

    # Error metrics
    mae = mean_absolute_error(golds, predictions)
    mse = mean_squared_error(golds, predictions)
    rmse = np.sqrt(mse)

    # R-squared (coefficient of determination)
    r2 = r2_score(golds, predictions)

    # Calculate correlation between probability and gold
    # Map gold ratings from [-3, 3] to [0, 1] for correlation with probability_true
    normalized_golds = (golds + 3) / 6
    correlation = np.corrcoef(probability_trues, normalized_golds)[0, 1]

    # Calculate confidence intervals using bootstrap
    # Combine predictions and golds for bootstrap sampling
    combined_data = np.column_stack((predictions, golds))

    # Define statistics functions for bootstrapping
    def bootstrap_accuracy(data):
        return np.mean(data[:, 0] == data[:, 1])

    def bootstrap_close_accuracy_1(data):
        return np.mean(np.abs(data[:, 0] - data[:, 1]) <= 1)

    def bootstrap_close_accuracy_2(data):
        return np.mean(np.abs(data[:, 0] - data[:, 1]) <= 2)

    def bootstrap_mae(data):
        return mean_absolute_error(data[:, 1], data[:, 0])

    # Calculate 95% confidence intervals
    accuracy_ci = bootstrap_ci(combined_data, bootstrap_accuracy)
    close_accuracy_1_ci = bootstrap_ci(combined_data, bootstrap_close_accuracy_1)
    close_accuracy_2_ci = bootstrap_ci(combined_data, bootstrap_close_accuracy_2)
    mae_ci = bootstrap_ci(combined_data, bootstrap_mae)

    # Calculate per-class metrics
    class_metrics = {}
    unique_golds = np.unique(golds)

    print("\nPer-class metrics (original scale -3 to +3):")
    for g in sorted(unique_golds):
        mask = (golds == g)
        if mask.sum() > 0:
            class_acc = np.mean(predictions[mask] == golds[mask])
            class_close_acc_1 = np.mean(np.abs(predictions[mask] - golds[mask]) <= 1)
            class_close_acc_2 = np.mean(np.abs(predictions[mask] - golds[mask]) <= 2)
            class_mae = mean_absolute_error(golds[mask], predictions[mask])
            class_avg_prob = np.mean(probability_trues[mask])
            count = mask.sum()

            print(f"  Class {g}: Acc={class_acc:.3f}, Close±1={class_close_acc_1:.3f}, "
                  f"Close±2={class_close_acc_2:.3f}, MAE={class_mae:.3f}, AvgProb={class_avg_prob:.3f} ({count} samples)")

            class_metrics[str(g)] = {
                "accuracy": float(class_acc),
                "close_accuracy_1": float(class_close_acc_1),
                "close_accuracy_2": float(class_close_acc_2),
                "mae": float(class_mae),
                "avg_probability": float(class_avg_prob),
                "count": int(count)
            }

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    possible_values = list(range(-3, 4))  # -3 to +3
    cm = confusion_matrix(
        golds,
        predictions,
        labels=possible_values
    )

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("True↓ Pred→ | " + " | ".join(f"{l:>3}" for l in possible_values))
    print("-" * (10 + 7 * len(possible_values)))
    for i, g in enumerate(possible_values):
        print(f"{g:>9} | " + " | ".join(f"{cm[i, j]:>3}" for j in range(len(possible_values))))

    # Print overall metrics with confidence intervals
    print(f"\nEvaluation metrics for {model_name}:")
    print(f"  Accuracy (exact match): {accuracy:.3f} [95% CI: {accuracy_ci[0]:.3f}-{accuracy_ci[1]:.3f}]")
    print(f"  Close Accuracy (within ±1): {close_accuracy_1:.3f} [95% CI: {close_accuracy_1_ci[0]:.3f}-{close_accuracy_1_ci[1]:.3f}]")
    print(f"  Close Accuracy (within ±2): {close_accuracy_2:.3f} [95% CI: {close_accuracy_2_ci[0]:.3f}-{close_accuracy_2_ci[1]:.3f}]")
    print(f"  Mean Absolute Error: {mae:.3f} [95% CI: {mae_ci[0]:.3f}-{mae_ci[1]:.3f}]")
    print(f"  Root Mean Squared Error: {rmse:.3f}")
    print(f"  R² (coefficient of determination): {r2:.3f}")
    print(f"  Correlation with gold ratings: {correlation:.3f}")

    # Return metrics as dictionary
    return {
        "accuracy": float(accuracy),
        "accuracy_ci_low": float(accuracy_ci[0]),
        "accuracy_ci_high": float(accuracy_ci[1]),
        "close_accuracy_1": float(close_accuracy_1),
        "close_accuracy_1_ci_low": float(close_accuracy_1_ci[0]),
        "close_accuracy_1_ci_high": float(close_accuracy_1_ci[1]),
        "close_accuracy_2": float(close_accuracy_2),
        "close_accuracy_2_ci_low": float(close_accuracy_2_ci[0]),
        "close_accuracy_2_ci_high": float(close_accuracy_2_ci[1]),
        "mae": float(mae),
        "mae_ci_low": float(mae_ci[0]),
        "mae_ci_high": float(mae_ci[1]),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "correlation": float(correlation),
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist()
    }

def calculate_predicate_metrics(all_data):
    """Calculate metrics grouped by predicate verb"""
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(all_data)

    # Group by verb and calculate metrics
    verb_metrics = {}

    for verb, group in df.groupby('Verb'):
        if len(group) < 5:  # Skip verbs with too few samples
            continue

        preds = np.array(group['pred'])
        golds = np.array(group['gold'])
        prob_trues = np.array(group['prob_true'])

        # Calculate metrics
        accuracy = np.mean(preds == golds)
        close_accuracy_1 = np.mean(np.abs(preds - golds) <= 1)
        close_accuracy_2 = np.mean(np.abs(preds - golds) <= 2)
        mae = mean_absolute_error(golds, preds)

        # Calculate correlation
        normalized_golds = (golds + 3) / 6
        correlation = np.corrcoef(prob_trues, normalized_golds)[0, 1] if len(prob_trues) > 1 else 0

        verb_metrics[verb] = {
            "count": len(group),
            "accuracy": float(accuracy),
            "close_accuracy_1": float(close_accuracy_1),
            "close_accuracy_2": float(close_accuracy_2),
            "mae": float(mae),
            "correlation": float(correlation)
        }

    # Print top and bottom 5 verbs by accuracy
    sorted_verbs = sorted(verb_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    print("\nTop 5 predicates by accuracy:")
    for verb, metrics in sorted_verbs[:5]:
        print(f"  {verb}: Acc={metrics['accuracy']:.3f}, Close±1={metrics['close_accuracy_1']:.3f}, Corr={metrics['correlation']:.3f}, Count={metrics['count']}")

    print("\nBottom 5 predicates by accuracy:")
    for verb, metrics in sorted_verbs[-5:]:
        print(f"  {verb}: Acc={metrics['accuracy']:.3f}, Close±1={metrics['close_accuracy_1']:.3f}, Corr={metrics['correlation']:.3f}, Count={metrics['count']}")

    return verb_metrics

def calculate_embedding_metrics(all_data):
    """Calculate metrics grouped by embedding environment"""
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(all_data)

    # Group by embedding and calculate metrics
    embedding_metrics = {}

    for embedding, group in df.groupby('Embedding'):
        if embedding == '' or len(group) < 5:  # Skip unknown embeddings or those with too few samples
            continue

        preds = np.array(group['pred'])
        golds = np.array(group['gold'])
        prob_trues = np.array(group['prob_true'])

        # Calculate metrics
        accuracy = np.mean(preds == golds)
        close_accuracy_1 = np.mean(np.abs(preds - golds) <= 1)
        close_accuracy_2 = np.mean(np.abs(preds - golds) <= 2)
        mae = mean_absolute_error(golds, preds)

        # Calculate correlation
        normalized_golds = (golds + 3) / 6
        correlation = np.corrcoef(prob_trues, normalized_golds)[0, 1] if len(prob_trues) > 1 else 0

        embedding_metrics[embedding] = {
            "count": len(group),
            "accuracy": float(accuracy),
            "close_accuracy_1": float(close_accuracy_1),
            "close_accuracy_2": float(close_accuracy_2),
            "mae": float(mae),
            "correlation": float(correlation)
        }

    # Print metrics by embedding
    print("\nMetrics by embedding environment:")
    for embedding, metrics in embedding_metrics.items():
        print(f"  {embedding}: Acc={metrics['accuracy']:.3f}, Close±1={metrics['close_accuracy_1']:.3f}, "
              f"Corr={metrics['correlation']:.3f}, Count={metrics['count']}")

    return embedding_metrics

def get_random_baseline_probabilities(row):
    """
    Generate random probability values from a normal distribution.
    Returns a value between 0 and 1 (clipped).
    """
    # Generate a random value from a normal distribution with mean 0.5, std 0.2
    # This gives a reasonable spread while keeping most values in the [0,1] range
    prob = np.random.normal(0.5, 0.2)
    return np.clip(prob, 0, 1)  # Clip to valid probability range

def evaluate_random_baseline(df, runs=50, max_samples=None, verbose=False):
    """
    Evaluates a random baseline on CommitmentBank by drawing random probabilities.
    Runs multiple times and averages the results.

    Parameters:
    - df: DataFrame containing CommitmentBank data
    - runs: Number of runs to average over
    - max_samples: Maximum number of samples to evaluate (None for all)
    - verbose: Whether to print progress for each run

    Returns: Dictionary of evaluation metrics (averaged) and prediction data
    """
    print(f"\nEvaluating random baseline ({runs} runs)...")

    # Store results from each run
    all_metrics = []
    all_prediction_data = []

    # Limit samples if specified
    if max_samples is not None:
        df_eval = df.sample(n=min(max_samples, len(df)), random_state=42)
    else:
        df_eval = df

    # Perform multiple runs
    for run in range(runs):
        if verbose:
            print(f"Run {run+1}/{runs}...")
        else:
            # Just show a progress counter without printing a new line each time
            if run % 10 == 0:
                print(f"Completed {run}/{runs} runs...", end="\r")

        # Process each item to get random probabilities
        all_data = []
        original_golds = []
        probability_trues = []

        for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), disable=not verbose):
            # Get random probability
            prob_true = get_random_baseline_probabilities(row)

            # Get gold rating from the 'mean.noTarget' column and round it
            gold_original = int(round(float(row["mean.noTarget"])))

            # Store results
            probability_trues.append(prob_true)
            original_golds.append(gold_original)

            # Store data
            all_data.append({
                'idx': idx,
                'uID': row.get('uID', ''),
                'Verb': row.get('Verb', ''),
                'Embedding': row.get('Embedding', ''),
                'Context': row.get('Context', ''),
                'Target': row.get('Target', ''),
                'Prompt': row.get('Prompt', ''),
                'gold': gold_original,
                'prob_true': prob_true
            })

        # Convert probabilities to numpy array for z-score calculation
        probability_trues = np.array(probability_trues)

        # Calculate z-scores for the probabilities
        prob_mean = np.mean(probability_trues)
        prob_std = np.std(probability_trues)

        if prob_std == 0:  # Handle edge case of zero standard deviation
            if verbose:
                print("Warning: Zero standard deviation in probabilities. Using fixed mapping.")
            z_scores = np.zeros_like(probability_trues)
        else:
            # Calculate z-scores (standardized scores)
            z_scores = (probability_trues - prob_mean) / prob_std

        # Use the CommitmentBank rating distribution boundaries
        bins = z_boundaries
        ratings = unique_ratings

        # For each z-score, find which bin it falls into
        predictions = np.zeros_like(z_scores, dtype=int)
        for i, z in enumerate(z_scores):
            for bin_idx in range(len(bins) - 1):
                if bins[bin_idx] <= z < bins[bin_idx + 1]:
                    predictions[i] = ratings[bin_idx]
                    break

        # Add predictions to the data
        for i, data_point in enumerate(all_data):
            data_point['pred'] = int(predictions[i])
            data_point['z_score'] = float(z_scores[i])
            data_point['error'] = abs(predictions[i] - data_point['gold'])

        # Convert to numpy arrays for metrics calculation
        original_predictions = np.array(predictions)
        original_golds = np.array(original_golds)

        # Calculate metrics for this run with a quieter setting
        metrics = calculate_metrics(original_predictions, original_golds, probability_trues, f"RandomBaseline_Run{run+1}")

        # Add z-score binning information to metrics
        metrics['z_score_bins'] = {
            'bins': bins.tolist(),
            'ratings': ratings.tolist(),
            'mean': float(prob_mean),
            'std_dev': float(prob_std),
            'gold_mean': float(gold_mean),
            'gold_std': float(gold_std)
        }

        all_metrics.append(metrics)
        all_prediction_data.append(all_data)

    print(f"\nCompleted all {runs} runs. Calculating average results...")

    # Average the metrics across all runs
    avg_metrics = {}

    # Get all the keys from the first run's metrics
    metric_keys = all_metrics[0].keys()

    # Collect all predictions and golds across runs for proper CI calculation
    all_predictions = []
    all_golds = []
    all_probability_trues = []

    # For each run
    for run_idx in range(runs):
        pred_data = all_prediction_data[run_idx]

        # Extract predictions and gold values
        run_preds = [item['pred'] for item in pred_data]
        run_golds = [item['gold'] for item in pred_data]
        run_probs = [item['prob_true'] for item in pred_data]

        all_predictions.extend(run_preds)
        all_golds.extend(run_golds)
        all_probability_trues.extend(run_probs)

    # Convert to numpy arrays for proper CI calculation
    all_predictions = np.array(all_predictions)
    all_golds = np.array(all_golds)
    all_probability_trues = np.array(all_probability_trues)

    # Combine predictions and golds for bootstrap sampling
    combined_data = np.column_stack((all_predictions, all_golds))

    # Define statistics functions for bootstrapping
    def bootstrap_accuracy(data):
        return np.mean(data[:, 0] == data[:, 1])

    def bootstrap_close_accuracy_1(data):
        return np.mean(np.abs(data[:, 0] - data[:, 1]) <= 1)

    def bootstrap_close_accuracy_2(data):
        return np.mean(np.abs(data[:, 0] - data[:, 1]) <= 2)

    def bootstrap_mae(data):
        return mean_absolute_error(data[:, 1], data[:, 0])

    # Calculate 95% confidence intervals - use more bootstraps for more precision
    accuracy_ci = bootstrap_ci(combined_data, bootstrap_accuracy, n_bootstraps=2000)
    close_accuracy_1_ci = bootstrap_ci(combined_data, bootstrap_close_accuracy_1, n_bootstraps=2000)
    close_accuracy_2_ci = bootstrap_ci(combined_data, bootstrap_close_accuracy_2, n_bootstraps=2000)
    mae_ci = bootstrap_ci(combined_data, bootstrap_mae, n_bootstraps=2000)

    # Add confidence intervals to metrics dictionary
    avg_metrics['accuracy_ci_low'] = float(accuracy_ci[0])
    avg_metrics['accuracy_ci_high'] = float(accuracy_ci[1])
    avg_metrics['close_accuracy_1_ci_low'] = float(close_accuracy_1_ci[0])
    avg_metrics['close_accuracy_1_ci_high'] = float(close_accuracy_1_ci[1])
    avg_metrics['close_accuracy_2_ci_low'] = float(close_accuracy_2_ci[0])
    avg_metrics['close_accuracy_2_ci_high'] = float(close_accuracy_2_ci[1])
    avg_metrics['mae_ci_low'] = float(mae_ci[0])
    avg_metrics['mae_ci_high'] = float(mae_ci[1])

    for key in metric_keys:
        if key in ['accuracy', 'close_accuracy_1', 'close_accuracy_2', 'mae', 'mse', 'rmse', 'r2', 'correlation']:
            # Average the simple numeric metrics
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
        elif key == 'class_metrics':
            # Average the class metrics
            avg_class_metrics = {}

            # Get all unique class values across all runs
            all_classes = set()
            for m in all_metrics:
                all_classes.update(m['class_metrics'].keys())

            for cls in all_classes:
                # Initialize with values from runs that have this class
                cls_metrics = [m['class_metrics'].get(cls, {}) for m in all_metrics if cls in m['class_metrics']]

                if cls_metrics:
                    avg_class_metrics[cls] = {}
                    for metric in ['accuracy', 'close_accuracy_1', 'close_accuracy_2', 'mae', 'avg_probability']:
                        values = [cm.get(metric, 0) for cm in cls_metrics]
                        avg_class_metrics[cls][metric] = float(np.mean(values))

                    # Use the average count
                    counts = [cm.get('count', 0) for cm in cls_metrics]
                    avg_class_metrics[cls]['count'] = int(np.mean(counts))

            avg_metrics['class_metrics'] = avg_class_metrics
        elif key == 'confusion_matrix':
            # Average the confusion matrices
            first_cm = np.array(all_metrics[0]['confusion_matrix'])
            avg_cm = np.zeros_like(first_cm, dtype=float)

            # Sum all confusion matrices
            for m in all_metrics:
                avg_cm += np.array(m['confusion_matrix'])

            # Divide by number of runs
            avg_cm = avg_cm / runs

            # Convert back to list of integers (rounded)
            avg_metrics['confusion_matrix'] = np.round(avg_cm).astype(int).tolist()
        elif key == 'z_score_bins':
            # Just use the values from the first run for reference
            avg_metrics['z_score_bins'] = all_metrics[0]['z_score_bins']

    # Print averaged results with confidence intervals
    print("\nAveraged Random Baseline Results (50 runs):")
    acc_ci = f"[{avg_metrics['accuracy_ci_low']:.3f}-{avg_metrics['accuracy_ci_high']:.3f}]"
    close1_ci = f"[{avg_metrics['close_accuracy_1_ci_low']:.3f}-{avg_metrics['close_accuracy_1_ci_high']:.3f}]"
    close2_ci = f"[{avg_metrics['close_accuracy_2_ci_low']:.3f}-{avg_metrics['close_accuracy_2_ci_high']:.3f}]"
    mae_ci = f"[{avg_metrics['mae_ci_low']:.3f}-{avg_metrics['mae_ci_high']:.3f}]"

    print(f"  Accuracy (exact match): {avg_metrics['accuracy']:.3f} {acc_ci}")
    print(f"  Close Accuracy (within ±1): {avg_metrics['close_accuracy_1']:.3f} {close1_ci}")
    print(f"  Close Accuracy (within ±2): {avg_metrics['close_accuracy_2']:.3f} {close2_ci}")
    print(f"  Mean Absolute Error: {avg_metrics['mae']:.3f} {mae_ci}")
    print(f"  Root Mean Squared Error: {avg_metrics['rmse']:.3f}")
    print(f"  R² (coefficient of determination): {avg_metrics['r2']:.3f}")
    print(f"  Correlation with gold ratings: {avg_metrics['correlation']:.3f}")

    # Return the averaged metrics and the prediction data from the last run
    return avg_metrics, all_prediction_data[-1]


if __name__ == "__main__":
    # Load CommitmentBank data
    df_clean = load_commitmentbank_data()

    # Define the models to evaluate
    model_names = {
        "Pythia-14M": "EleutherAI/pythia-14m",
        "Pythia-70M": "EleutherAI/pythia-70m",
        "Pythia-160M": "EleutherAI/pythia-160m",
        "Pythia-410M": "EleutherAI/pythia-410m",
        "Pythia-1b": "EleutherAI/pythia-1b",
        "OLMo-1B": "allenai/OLMo-1B-0724-hf",
        "Llama-3.2-1B": "meta-llama/Llama-3.2-1B"
    }

    # Setting max_samples to None will run on ALL samples
    # You can set a smaller number like 200 for faster testing
    max_samples = None  # Set to None for full evaluation, or a number for testing

    # Run the full evaluation on all models
    results = {}
    predictions = {}

    # First evaluate the random baseline
    print(f"\n===== Evaluating Random Baseline on CommitmentBank =====")
    baseline_metrics, baseline_predictions = evaluate_random_baseline(
        df_clean,
        runs=50,  # Run 50 iterations
        max_samples=max_samples
    )
    results["Random Baseline"] = baseline_metrics
    predictions["Random Baseline"] = baseline_predictions

    # Then evaluate the models
    for display_name, model_name in model_names.items():
        print(f"\n===== Evaluating {display_name} on CommitmentBank =====")

        # Run evaluation
        metrics, prediction_data = evaluate_model(
            model_name,
            df_clean,
            save_predictions=True,
            prediction_file=f"{display_name}_predictions.csv",
            max_samples=max_samples
        )

        # Store results
        results[display_name] = metrics
        predictions[display_name] = prediction_data

    # Print a comprehensive summary of the results with confidence intervals
    print("\n======= Comprehensive Results Summary =======")
    print("Model           | Accuracy (95% CI)         | Close±1 (95% CI)          | Close±2 (95% CI)          | MAE (95% CI)              ")
    print("--------------- | -------------------------- | -------------------------- | -------------------------- | --------------------------")
    for display_name, metrics in results.items():
        # Check if confidence intervals are present in the metrics
        # If not, use default values
        acc_ci_low = metrics.get('accuracy_ci_low', metrics.get('accuracy') - 0.05)
        acc_ci_high = metrics.get('accuracy_ci_high', metrics.get('accuracy') + 0.05)
        close1_ci_low = metrics.get('close_accuracy_1_ci_low', metrics.get('close_accuracy_1') - 0.05)
        close1_ci_high = metrics.get('close_accuracy_1_ci_high', metrics.get('close_accuracy_1') + 0.05)
        close2_ci_low = metrics.get('close_accuracy_2_ci_low', metrics.get('close_accuracy_2') - 0.05)
        close2_ci_high = metrics.get('close_accuracy_2_ci_high', metrics.get('close_accuracy_2') + 0.05)
        mae_ci_low = metrics.get('mae_ci_low', metrics.get('mae') * 0.95)
        mae_ci_high = metrics.get('mae_ci_high', metrics.get('mae') * 1.05)

        acc_ci = f"[{acc_ci_low:.3f}-{acc_ci_high:.3f}]"
        close1_ci = f"[{close1_ci_low:.3f}-{close1_ci_high:.3f}]"
        close2_ci = f"[{close2_ci_low:.3f}-{close2_ci_high:.3f}]"
        mae_ci = f"[{mae_ci_low:.3f}-{mae_ci_high:.3f}]"

        print(f"{display_name:<15} | {metrics['accuracy']:.3f} {acc_ci:<16} | "
              f"{metrics['close_accuracy_1']:.3f} {close1_ci:<16} | "
              f"{metrics['close_accuracy_2']:.3f} {close2_ci:<16} | "
              f"{metrics['mae']:.3f} {mae_ci:<16}")

    # Save the complete results
    with open("commitmentbank_distribution_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model, metrics in results.items():
            json_results[model] = {k: v for k, v in metrics.items()}

        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to commitmentbank_distribution_results.json")