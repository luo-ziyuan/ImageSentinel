import json
import numpy as np
from sklearn import metrics
import random
import os
import argparse
from pathlib import Path
from scipy import stats

def load_similarities(json_path):
    """Load similarity file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
        # Return both the full data and simplified similarity data
        similarities = {k: float(v['similarity']) for k, v in data.items()}
        return data, similarities

def check_retrieval_accuracy(data):
    """Calculate retrieval accuracy metrics"""
    total = len(data)
    metrics = {
        'Hit@1': 0,     # Is the first result correct
        'Hit@3': 0,     # Is the correct answer in the top 3 results
        'Hit@5': 0,     # Is the correct answer in the top 5 results
        'Hit@10': 0,    # Is the correct answer in the top 10 results
        'Total_Queries': total
    }
    
    for img_name, item in data.items():
        # Get the image number (remove the .jpg suffix)
        img_num = img_name.replace('.jpg', '')
        sentinel_pattern = f"sentinel_{img_num}"
        
        # Iterate over the retrieved results list
        retrieved_list = item['retrieved_images']
        
        # Find the position of the correct answer
        for rank, retrieved in enumerate(retrieved_list, 1):
            if sentinel_pattern in retrieved:
                # Update Hit@k metrics
                if rank <= 1:
                    metrics['Hit@1'] += 1
                if rank <= 3:
                    metrics['Hit@3'] += 1
                if rank <= 5:
                    metrics['Hit@5'] += 1
                if rank <= 10:
                    metrics['Hit@10'] += 1
                break
    
    # Normalize metrics
    metrics['Hit@1'] /= total
    metrics['Hit@3'] /= total
    metrics['Hit@5'] /= total
    metrics['Hit@10'] /= total
    
    return metrics

def calculate_metrics(sentinel_values, original_values):
    """Calculate ROC-AUC and TPR@FPR metrics"""
    # Combine predictions and labels
    preds = original_values + sentinel_values  # Higher similarity indicates the result is more likely to be sentinel (positive class)
    labels = [0] * len(original_values) + [1] * len(sentinel_values)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    # Calculate TPR@1%FPR and TPR@10%FPR
    tpr_at_1_fpr = np.interp(0.01, fpr, tpr)
    tpr_at_10_fpr = np.interp(0.10, fpr, tpr)
    
    return {
        'ROC-AUC': auc,
        'TPR@1%FPR': tpr_at_1_fpr,
        'TPR@10%FPR': tpr_at_10_fpr
    }

def calculate_metrics_with_error(all_runs_results):
    """Calculate mean and various error metrics"""
    metrics = all_runs_results[0].keys()
    results = {}
    
    for metric in metrics:
        values = [run[metric] for run in all_runs_results]
        mean = np.mean(values)
        
        results[metric] = {
            'mean': mean,
        }
    
    return results

def sample_and_evaluate(sentinel_out_dir, original_dir, similarity_type, num_samples, num_times, total_runs):
    """Main evaluation function"""
    # Build file paths
    sentinel_json = Path(sentinel_out_dir) / f'{similarity_type}_similarities.json'
    original_json = Path(original_dir) / f'{similarity_type}_similarities.json'
    
    # Load data
    sentinel_full_data, sentinel_similarities = load_similarities(sentinel_json)
    original_full_data, original_similarities = load_similarities(original_json)
    
    # Calculate retrieval accuracy
    sentinel_accuracy = check_retrieval_accuracy(sentinel_full_data)
    
    # Get common keys
    common_keys = list(set(sentinel_similarities.keys()) & set(original_similarities.keys()))
    
    if len(common_keys) < num_samples:
        raise ValueError(f"Not enough common samples. Required: {num_samples}, Available: {len(common_keys)}")
    
    # Store results for all runs
    all_runs_results = []
    
    for run in range(total_runs):
        print(f"Running evaluation {run + 1}/{total_runs}")
        
        # Store sampled results for the current run
        sentinel_values_list = []
        original_values_list = []
        
        # Perform multiple samples
        for _ in range(num_times):
            # Randomly select samples
            selected_keys = random.sample(common_keys, num_samples)
            
            # Calculate average similarity
            sentinel_avg = np.mean([sentinel_similarities[k] for k in selected_keys])
            original_avg = np.mean([original_similarities[k] for k in selected_keys])
            
            sentinel_values_list.append(sentinel_avg)
            original_values_list.append(original_avg)
        
        # Calculate metrics for the current run
        current_metrics = calculate_metrics(sentinel_values_list, original_values_list)
        current_metrics.update({
            'Sentinel_' + k: v for k, v in sentinel_accuracy.items()
        })
        all_runs_results.append(current_metrics)
        
    # Calculate overall statistics for all runs
    metrics_stats = calculate_metrics_with_error(all_runs_results)
    
    # Save results
    output_path = Path(sentinel_out_dir) / f'{similarity_type}_evaluation_results_samples{num_samples}_times{num_times}_totalruns{total_runs}.json'
    
    results = {
        'parameters': {
            'num_samples': num_samples,
            'num_times': num_times,
            'total_runs': total_runs,
            'similarity_type': similarity_type
        },
        'metrics_statistics': metrics_stats,
        'all_runs_results': all_runs_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_path}")
    
    return metrics_stats

def main():
    parser = argparse.ArgumentParser(description='Evaluate similarity metrics')
    parser.add_argument('--sentinel_out_dir', type=str, required=True, help='Directory containing sentinel results')
    parser.add_argument('--original_dir', type=str, required=True, help='Directory containing original results')
    parser.add_argument('--similarity_type', type=str, required=True, choices=['clip', 'dino', 'dinov2', 'simsiam', 'hidden', 'fin'],
                        help='Type of similarity metric to evaluate')
    parser.add_argument('--num_samples', type=int, required=True, help='Number of samples to use in each evaluation')
    parser.add_argument('--num_times', type=int, default=100, help='Number of times to repeat the evaluation')
    parser.add_argument('--total_runs', type=int, default=5, help='Total number of complete evaluation runs')
    
    args = parser.parse_args()
    
    try:
        metrics_stats = sample_and_evaluate(
            args.sentinel_out_dir,
            args.original_dir,
            args.similarity_type,
            args.num_samples,
            args.num_times,
            args.total_runs
        )
        
        print("\nFinal Results:")
        for metric, stats in metrics_stats.items():
            print(f"\n{metric}:")
            print(f"  {stats['mean']:.4f}")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()