import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
import random
import time
import psutil
import gc

# Set up argument parser
parser = argparse.ArgumentParser(description='Train rule embeddings with different random seeds')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--output_dir', type=str, default='output/results_rule_b', help='Directory to save results')
parser.add_argument('--obs_prob', type=float, default=0.3, help='Probability of observing hidden variables')
parser.add_argument('--n_samples', type=int, default=20000, help='Number of samples to generate')
parser.add_argument('--epochs_per_block', type=int, default=30, help='Number of epochs for each training block (X3, X4, or X5)')
parser.add_argument('--max_cycles', type=int, default=5, help='Maximum number of coordinate descent cycles')
parser.add_argument('--missing_mechanism', type=str, default='MCAR', choices=['MCAR', 'MAR', 'MNAR'], 
                   help='Missing data mechanism: MCAR (Missing Completely At Random), MAR (Missing At Random), MNAR (Missing Not At Random)')
parser.add_argument('--mar_dependency', type=str, default='X0', 
                   help='For MAR: which observed variable determines missingness (X0, X1, X2, X6, X7)')
parser.add_argument('--mnar_threshold', type=float, default=0.5, 
                   help='For MNAR: threshold for missingness based on true values')
# Temperature / beta settings
parser.add_argument('--beta_mode', type=str, default='constant', choices=['constant', 'cycle'],
                   help='How to set temperature beta for X5 OR learning: constant or cosine-decay cycle')
parser.add_argument('--beta_const', type=float, default=10.0,
                   help='Constant beta value when beta_mode=constant')
parser.add_argument('--beta_start', type=float, default=10.0,
                   help='Starting beta when beta_mode=cycle')
parser.add_argument('--beta_stop', type=float, default=30.0,
                   help='Final beta when beta_mode=cycle')
parser.add_argument('--beta_n_cycle', type=int, default=4,
                   help='Number of cosine cycles for beta annealing when beta_mode=cycle')
parser.add_argument('--beta_ratio', type=float, default=0.7,
                   help='Upward proportion in each cosine cycle when beta_mode=cycle')
parser.add_argument('--beta_decay', type=float, default=0.7,
                   help='Multiplicative decay factor for beta across cycles when beta_mode=cycle')
parser.add_argument('--softmin_temp', type=float, default=0.1,
                   help='Temperature for softmin when inferring conjunctive clauses')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def frange_cycle_cosine_decay(n_iter, start, stop, n_cycle, ratio, decay):
    L = np.zeros(n_iter)
    period = n_iter / n_cycle
    for c in range(n_cycle):
        min_val = start * (decay ** c)
        max_val = stop * (decay ** c)
        cycle_start = int(c * period)
        cycle_end = int((c + 1) * period)
        up_len = int(period * ratio)
        down_len = int(period - up_len)
        if up_len > 0:
            up_indices = np.arange(cycle_start, min(cycle_start + up_len, n_iter))
            if len(up_indices) > 0:
                phase = np.linspace(0, np.pi/2, len(up_indices))
                L[up_indices] = min_val + (max_val - min_val) * np.sin(phase)
        if down_len > 0 and cycle_start + up_len < n_iter:
            down_indices = np.arange(cycle_start + up_len, min(cycle_end, n_iter))
            if len(down_indices) > 0:
                phase = np.linspace(np.pi/2, np.pi, len(down_indices))
                L[down_indices] = min_val + (max_val - min_val) * np.sin(phase)
    return L

def generate_data(n_samples, p_X0, p_X1, p_X2, p_X6, p_X7, obs_prob, missing_mechanism='MCAR', mar_dependency='X0', mnar_threshold=0.5):
    """
    Generate data with different missing data mechanisms:
    - MCAR: Missing Completely At Random (independent of any variables)
    - MAR: Missing At Random (depends on observed variables)
    - MNAR: Missing Not At Random (depends on the true values themselves)
    """
    # Generate base variables
    X0 = torch.bernoulli(torch.full((n_samples,), p_X0))
    X1 = torch.bernoulli(torch.full((n_samples,), p_X1))
    X2 = torch.bernoulli(torch.full((n_samples,), p_X2))
    X6 = torch.bernoulli(torch.full((n_samples,), p_X6))
    X7 = torch.bernoulli(torch.full((n_samples,), p_X7))

    # Generate hidden predicates with multiple rules
    X3 = torch.where((X0 == 1) & (X1 == 1), torch.tensor(1.0), torch.tensor(0.0))
    X4 = torch.where((X2 == 1) & (X7 == 1), torch.tensor(1.0), torch.tensor(0.0))

    # X5 can be triggered by two rules
    X5_rule1 = torch.where((X3 == 1) & (X6 == 1), torch.tensor(1.0), torch.tensor(0.0))
    X5_rule2 = torch.where((X4 == 1) & (X0 == 1), torch.tensor(1.0), torch.tensor(0.0))
    X5 = torch.where((X5_rule1 == 1) | (X5_rule2 == 1), torch.tensor(1.0), torch.tensor(0.0))
    
    # Create observation masks based on missing mechanism
    if missing_mechanism == 'MCAR':
        # Missing Completely At Random - independent of any variables
        obs_mask = torch.bernoulli(torch.full((n_samples, 3), obs_prob))  # For X3, X4, X5
        
    elif missing_mechanism == 'MAR':
        # Missing At Random - depends on observed variables
        obs_mask = torch.zeros((n_samples, 3))
        
        # Determine which observed variable to use for missingness
        if mar_dependency == 'X0':
            dependency_var = X0
        elif mar_dependency == 'X1':
            dependency_var = X1
        elif mar_dependency == 'X2':
            dependency_var = X2
        elif mar_dependency == 'X6':
            dependency_var = X6
        elif mar_dependency == 'X7':
            dependency_var = X7
        else:
            dependency_var = X0  # Default to X0
            
        # Higher probability of missing when dependency variable is 1
        # This creates a bias where certain patterns are more likely to be missing
        for i in range(3):
            # When dependency_var=1, higher chance of missing (lower obs_prob)
            # When dependency_var=0, lower chance of missing (higher obs_prob)
            adjusted_prob = obs_prob * (1.0 - 0.3 * dependency_var)  # Adjust based on dependency
            adjusted_prob = torch.clamp(adjusted_prob, 0.1, 0.9)  # Keep reasonable bounds
            obs_mask[:, i] = torch.bernoulli(adjusted_prob)
            
    elif missing_mechanism == 'MNAR':
        # Missing Not At Random - depends on the true values themselves
        obs_mask = torch.zeros((n_samples, 3))
        
        # For X3: more likely to be missing when X3=1 (positive values are harder to observe)
        prob_X3_missing = torch.where(X3 == 1, obs_prob * 0.3, obs_prob * 1.5)  # Bias against positive values
        prob_X3_missing = torch.clamp(prob_X3_missing, 0.1, 0.9)
        obs_mask[:, 0] = torch.bernoulli(prob_X3_missing)
        
        # For X4: more likely to be missing when X4=1
        prob_X4_missing = torch.where(X4 == 1, obs_prob * 0.3, obs_prob * 1.5)
        prob_X4_missing = torch.clamp(prob_X4_missing, 0.1, 0.9)
        obs_mask[:, 1] = torch.bernoulli(prob_X4_missing)
        
        # For X5: more likely to be missing when X5=1
        prob_X5_missing = torch.where(X5 == 1, obs_prob * 0.3, obs_prob * 1.5)
        prob_X5_missing = torch.clamp(prob_X5_missing, 0.1, 0.9)
        obs_mask[:, 2] = torch.bernoulli(prob_X5_missing)
        
    else:
        raise ValueError(f"Unknown missing mechanism: {missing_mechanism}")
    
    # Create observed values (NaN for unobserved)
    X3_obs = torch.where(obs_mask[:, 0] == 1, X3, torch.tensor(float('nan')))
    X4_obs = torch.where(obs_mask[:, 1] == 1, X4, torch.tensor(float('nan')))
    X5_obs = torch.where(obs_mask[:, 2] == 1, X5, torch.tensor(float('nan')))
    
    # Stack all data including observed values and masks
    data_tensor = torch.stack([
        X0, X1, X2, X6, X7,         # 0-4: Base variables
        X3_obs, X4_obs, X5_obs,     # 5-7: Observed hidden predicates
        obs_mask[:, 0], obs_mask[:, 1], obs_mask[:, 2],  # 8-10: Observation masks
        X3, X4, X5                  # 11-13: True values for accuracy calculation
        # X5_rule1, X5_rule2        # 14-15: Rule masks for X5
    ], dim=1).float()

    return data_tensor


predicate_embeddings = {
    'X0': torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    'X1': torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    'X2': torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
    'X3': torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32),
    'X4': torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32),
    'X5': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32),
    'X6': torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32),
    'X7': torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
}

predicate_embeddings = {key: value / torch.norm(value, p=2) for key, value in predicate_embeddings.items()}  # Normalize embeddings

emb_dim = 8

rule1_embedding = torch.rand(2, emb_dim, requires_grad=True)
rule2_embedding = torch.rand(2, emb_dim, requires_grad=True)
rule3_embedding = torch.rand(2, emb_dim, requires_grad=True)
rule4_embedding = torch.rand(2, emb_dim, requires_grad=True)

# print("initial rule embeddings:")
# print("rule1_embedding: ", rule1_embedding)
# print("rule2_embedding: ", rule2_embedding)
# print("rule3_embedding: ", rule3_embedding)
# print("rule4_embedding: ", rule4_embedding)


def cosine_similarity(a, b):
    cosine_sim = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    noise = torch.normal(0.0, 0.1, size=(1,))
    return cosine_sim + noise.item()

def select_predicates_for_rule_X(rule_embedding, predicate_embeddings):
    event_names = ['X0', 'X1', 'X2', 'X6', 'X7']
    selected_indices = []
    for rule_vec in rule_embedding:
        similarities = torch.tensor([cosine_similarity(rule_vec, predicate_embeddings[event]) for event in event_names])
        sorted_indices = torch.argsort(similarities)
        selected_indices.append((event_names[sorted_indices[-2].item()], event_names[sorted_indices[-1].item()]))
    return selected_indices

def select_predicates_for_rule_Y(rule_embedding, predicate_embeddings):
    event_names = ['X0', 'X1', 'X2', 'X3', 'X4', 'X6', 'X7']
    selected_indices = []
    for rule_vec in rule_embedding:
        similarities = torch.tensor([cosine_similarity(rule_vec, predicate_embeddings[event]) for event in event_names])
        sorted_indices = torch.argsort(similarities)
        selected_indices.append((event_names[sorted_indices[-2].item()], event_names[sorted_indices[-1].item()]))
    return selected_indices


def softmin(values, temperature=10.0):
    """Compute softmin of tensor values"""
    # values should be a tensor of shape (batch_size, n_values)
    scaled = -values / temperature
    weights = F.softmax(scaled, dim=1)  # Softmax along the values dimension
    return torch.sum(weights * values, dim=1)  # Sum along the values dimension


def infer_X(data_batch, softmin_temp):
    # Get base variables
    v0 = torch.where(data_batch[:, 0] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v1 = torch.where(data_batch[:, 1] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v2 = torch.where(data_batch[:, 2] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v6 = torch.where(data_batch[:, 3] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v7 = torch.where(data_batch[:, 4] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v5 = torch.tensor(0.5).repeat(len(data_batch))
    v3 = torch.tensor(0.5).repeat(len(data_batch))
    v4 = torch.tensor(0.5).repeat(len(data_batch))

    X3_obs = data_batch[:, 5]
    X4_obs = data_batch[:, 6]
    mask_X3 = data_batch[:, 8]
    mask_X4 = data_batch[:, 9]

    event_embeddings = {'X0': v0, 'X1': v1, 'X2': v2, 'X3': v3, 'X4': v4, 'X5': v5, 'X6': v6, 'X7': v7}

    # Get selected predicates for rules
    selected_indices_X3 = select_predicates_for_rule_X(rule1_embedding, predicate_embeddings)
    selected_indices_X4 = select_predicates_for_rule_X(rule2_embedding, predicate_embeddings)

    # Calculate cosine similarities
    cosX3_1 = cosine_similarity(rule1_embedding[0], predicate_embeddings[selected_indices_X3[0][1]])
    cosX3_2 = cosine_similarity(rule1_embedding[1], predicate_embeddings[selected_indices_X3[1][1]])

    cosX4_1 = cosine_similarity(rule2_embedding[0], predicate_embeddings[selected_indices_X4[0][1]])
    cosX4_2 = cosine_similarity(rule2_embedding[1], predicate_embeddings[selected_indices_X4[1][1]])

    # Calculate inferred values using softmin
    X3_values = torch.stack([
            cosX3_1 * torch.ones_like(v0),
            cosX3_2 * torch.ones_like(v0),
            event_embeddings[selected_indices_X3[0][1]],
            event_embeddings[selected_indices_X3[1][1]]
        ], dim=1)
    
    X4_values = torch.stack([
        cosX4_1 * torch.ones_like(v0),
        cosX4_2 * torch.ones_like(v0),
        event_embeddings[selected_indices_X4[0][1]],
        event_embeddings[selected_indices_X4[1][1]]
    ], dim=1)

    inferred_v3 = softmin(X3_values, temperature=softmin_temp)
    inferred_v4 = softmin(X4_values, temperature=softmin_temp)

    return inferred_v3, inferred_v4


def lse_softmax(values, beta, return_probs=False):
    # values: tensor of shape (n,) or (batch, n)
    # beta: temperature parameter
    stacked = torch.stack(values, dim=0)
    logits = beta * stacked
    probs = F.softmax(logits, dim=0)

    lse = (1.0 / beta) * torch.logsumexp(logits, dim=0)

    if return_probs:
        return lse, probs
    return lse

def infer_v5(data_batch, beta, softmin_temp, return_probs=False, rule_idx=None):
    v0 = torch.where(data_batch[:, 0] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v1 = torch.where(data_batch[:, 1] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v2 = torch.where(data_batch[:, 2] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v6 = torch.where(data_batch[:, 3] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v7 = torch.where(data_batch[:, 4] == 1, torch.tensor(1.0), torch.tensor(0.0))
    v5 = torch.tensor(0.5).repeat(len(data_batch))
    v3, v4 = infer_X(data_batch, softmin_temp=softmin_temp)
    X3_obs = data_batch[:, 5]
    X4_obs = data_batch[:, 6]
    valid_X3 = ~torch.isnan(X3_obs)
    valid_X4 = ~torch.isnan(X4_obs)
    v3[valid_X3] = X3_obs[valid_X3]
    v4[valid_X4] = X4_obs[valid_X4]
    event_embeddings = {'X0': v0, 'X1': v1, 'X2': v2, 'X3': v3, 'X4': v4, 'X5': v5, 'X6': v6, 'X7': v7}
    
    # Use rule3_embedding for the first rule
    selected_indices = select_predicates_for_rule_Y(rule3_embedding, predicate_embeddings)
    cos1 = cosine_similarity(rule3_embedding[0], predicate_embeddings[selected_indices[0][1]])
    cos2 = cosine_similarity(rule3_embedding[1], predicate_embeddings[selected_indices[1][1]])
    
    # Use rule4_embedding for the second rule
    selected_indices_2 = select_predicates_for_rule_Y(rule4_embedding, predicate_embeddings)
    cos2_1 = cosine_similarity(rule4_embedding[0], predicate_embeddings[selected_indices_2[0][1]])
    cos2_2 = cosine_similarity(rule4_embedding[1], predicate_embeddings[selected_indices_2[1][1]])
    
    # Stack values for softmin for first rule
    X5_rule1_values = torch.stack([
        cos1 * torch.ones_like(v0),
        cos2 * torch.ones_like(v0),
        event_embeddings[selected_indices[0][1]],
        event_embeddings[selected_indices[1][1]]
    ], dim=1)
    
    # Stack values for softmin for second rule
    X5_rule2_values = torch.stack([
        cos2_1 * torch.ones_like(v0),
        cos2_2 * torch.ones_like(v0),
        event_embeddings[selected_indices_2[0][1]],
        event_embeddings[selected_indices_2[1][1]]
    ], dim=1)
    
    # Calculate inferred values using softmin
    v5_1 = softmin(X5_rule1_values, temperature=softmin_temp)
    v5_2 = softmin(X5_rule2_values, temperature=softmin_temp)

    if rule_idx == 0:
        # When training rule 1, make sure gradients only flow to rule3_embedding
        with torch.no_grad():
            v5_2 = softmin(X5_rule2_values, temperature=softmin_temp)
        return v5_1
    elif rule_idx == 1:
        # When training rule 2, make sure gradients only flow to rule4_embedding
        with torch.no_grad():
            v5_1 = softmin(X5_rule1_values, temperature=softmin_temp)
        return v5_2
    else:
        return lse_softmax([v5_1, v5_2], beta, return_probs=return_probs)


def loss_function(data_batch, beta, softmin_temp, lambda_l1=0.00, entropy_weight=0.0):
    v3, v4 = infer_X(data_batch, softmin_temp=softmin_temp)
    X3_obs = data_batch[:, 5]
    X4_obs = data_batch[:, 6]
    X5_obs = data_batch[:, 7]
    mask_X3 = data_batch[:, 8]
    mask_X4 = data_batch[:, 9]
    mask_X5 = data_batch[:, 10]
    
    # Calculate MSE loss for each observed predicate, ignoring NaN values
    valid_X3 = ~torch.isnan(X3_obs)
    valid_X4 = ~torch.isnan(X4_obs)
    valid_X5 = ~torch.isnan(X5_obs)
    
    loss_X3 = torch.mean((v3[valid_X3] - X3_obs[valid_X3]) ** 2) if torch.any(valid_X3) else torch.tensor(0.0)
    loss_X4 = torch.mean((v4[valid_X4] - X4_obs[valid_X4]) ** 2) if torch.any(valid_X4) else torch.tensor(0.0)
    v5 = infer_v5(data_batch, beta, softmin_temp=softmin_temp)
    loss_X5 = torch.mean((v5[valid_X5] - X5_obs[valid_X5]) ** 2) if torch.any(valid_X5) else torch.tensor(0.0)
    # Entropy loss (encourage one-hot)
    entropy = -torch.sum(v5 * torch.log(v5 + 1e-8), dim=0)
    mean_entropy = torch.mean(entropy)
    l1_penalty = lambda_l1 * (torch.sum(torch.abs(rule1_embedding)) + 
                             torch.sum(torch.abs(rule2_embedding)) + 
                             torch.sum(torch.abs(rule3_embedding)))
    total_loss = loss_X3 + loss_X4 + loss_X5 + entropy_weight * mean_entropy + l1_penalty
    return total_loss


def normalize_embeddings():
    with torch.no_grad():
        rule1_embedding.data = rule1_embedding.data / torch.norm(rule1_embedding.data, dim=1, keepdim=True)
        rule2_embedding.data = rule2_embedding.data / torch.norm(rule2_embedding.data, dim=1, keepdim=True)
        rule3_embedding.data = rule3_embedding.data / torch.norm(rule3_embedding.data, dim=1, keepdim=True)
        rule4_embedding.data = rule4_embedding.data / torch.norm(rule4_embedding.data, dim=1, keepdim=True)


def accuracy(data_batch, beta, softmin_temp): # input batch
    v5 = infer_v5(data_batch, beta, softmin_temp=softmin_temp)
    predicted = torch.where(v5 > 0.5, torch.tensor(1.0), torch.tensor(0.0))
    labels = data_batch[:, -1].float()
    correct = torch.sum(predicted == labels).item()
    return correct / len(data_batch)

def analyze_rules():
    print("\nAnalyzing learned rules:")
    
    # Analyze rule1_embedding (for X3)
    # print("\nRule for X3:")
    selected_indices_X3 = select_predicates_for_rule_X(rule1_embedding, predicate_embeddings)
    # print(f"Rule embedding: {rule1_embedding.data}")

    # Analyze rule2_embedding (for X4)
    # print("\nRule for X4:")
    selected_indices_X4 = select_predicates_for_rule_X(rule2_embedding, predicate_embeddings)
    # print(f"Rule embedding: {rule2_embedding.data}")
    
    # Analyze rule3_embedding (for X5)
    # print("\nRule for X5:")
    selected_indices_X5 = select_predicates_for_rule_Y(rule3_embedding, predicate_embeddings)
    # print(f"Rule embedding: {rule3_embedding.data}")

    # Analyze rule4_embedding (for X5)
    # print("\nRule for X5:")
    selected_indices_X5_2 = select_predicates_for_rule_Y(rule4_embedding, predicate_embeddings)
    # print(f"Rule embedding: {rule4_embedding.data}")

    # Print the actual rule structure
    print("\nRecovered Rule Structure:")
    print(f"X3 = {selected_indices_X3[0][1]} AND {selected_indices_X3[1][1]}")
    print(f"X4 = {selected_indices_X4[0][1]} AND {selected_indices_X4[1][1]}")
    print(f"X5 = {selected_indices_X5[0][1]} AND {selected_indices_X5[1][1]} OR {selected_indices_X5_2[0][1]} AND {selected_indices_X5_2[1][1]}")

    # Save results to json file, include beta and softmin_temp settings in filename to avoid overwrite
    if args.beta_mode == "constant":
        beta_tag = f"const{args.beta_const}"
    else:
        beta_tag = f"cycle{args.beta_start}-{args.beta_stop}"
    softmin_tag = f"smt{args.softmin_temp}"
    results = {
        "seed": args.seed,
        "X3": {
            "rule_embedding": rule1_embedding.data.tolist(),
            "selected_predicates": [selected_indices_X3[0][1], selected_indices_X3[1][1]]
        },
        "X4": {
            "rule_embedding": rule2_embedding.data.tolist(),
            "selected_predicates": [selected_indices_X4[0][1], selected_indices_X4[1][1]]
        },
        "X5_Rule1": {
            "rule_embedding": rule3_embedding.data.tolist(),
            "selected_predicates": [selected_indices_X5[0][1], selected_indices_X5[1][1]]
        },
        "X5_Rule2": {
            "rule_embedding": rule4_embedding.data.tolist(),
            "selected_predicates": [selected_indices_X5_2[0][1], selected_indices_X5_2[1][1]]
        },
        "recovered_structure": {
            "X3": f"{selected_indices_X3[0][1]} AND {selected_indices_X3[1][1]}",
            "X4": f"{selected_indices_X4[0][1]} AND {selected_indices_X4[1][1]}",
            "X5": f"{selected_indices_X5[0][1]} AND {selected_indices_X5[1][1]} OR {selected_indices_X5_2[0][1]} AND {selected_indices_X5_2[1][1]}"
        }
    }
    
    json_filename = os.path.join(
        args.output_dir,
        f'rule_analysis_seed_{args.seed}_obs{args.obs_prob}_n{args.n_samples}_beta_{beta_tag}_{softmin_tag}.json'
    )
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    # print(f"Results saved to {json_filename}")

def calculate_unobserved_accuracy(data_batch, beta, softmin_temp):
    # Get predictions
    v3, v4 = infer_X(data_batch, softmin_temp=softmin_temp)
    v5 = infer_v5(data_batch, beta, softmin_temp=softmin_temp)
    
    # Get masks and true values
    mask_X3 = data_batch[:, 8]
    mask_X4 = data_batch[:, 9]
    mask_X5 = data_batch[:, 10]
    true_X3 = data_batch[:, 11]
    true_X4 = data_batch[:, 12]
    true_X5 = data_batch[:, 13]
    
    # Convert predictions to binary (0/1)
    pred_X3 = (v3 > 0.5).float()
    pred_X4 = (v4 > 0.5).float()
    pred_X5 = (v5 > 0.5).float()
    
    # Calculate accuracy for unobserved values
    unobs_X3 = (mask_X3 == 0)
    unobs_X4 = (mask_X4 == 0)
    unobs_X5 = (mask_X5 == 0)
    
    acc_X3 = torch.mean((pred_X3[unobs_X3] == true_X3[unobs_X3]).float()) if torch.any(unobs_X3) else torch.tensor(0.0)
    acc_X4 = torch.mean((pred_X4[unobs_X4] == true_X4[unobs_X4]).float()) if torch.any(unobs_X4) else torch.tensor(0.0)
    acc_X5 = torch.mean((pred_X5[unobs_X5] == true_X5[unobs_X5]).float()) if torch.any(unobs_X5) else torch.tensor(0.0)
    
    return acc_X3.item(), acc_X4.item(), acc_X5.item()


def get_subsequence_lengths(data_list):
    if not data_list:
        return []

    zero_indices = [i for i, x in enumerate(data_list) if x == 0]

    if not zero_indices:
        return []

    lengths = [0]
    for k in range(len(zero_indices)):
        start_index = zero_indices[k]
        
        if k + 1 < len(zero_indices):
            end_index = zero_indices[k+1]
        else:
            end_index = len(data_list)
        
        lengths.append(end_index - start_index)
        
    return lengths


def cleanup_memory():
    """Clean up memory and release unnecessary caches"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(data_set, batch_size, epochs_per_block, max_cycles): # Modified signature
    # Start timing and memory tracking
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Initialize optimizers for each rule (once, before cycles)
    optimizer_X3 = torch.optim.Adam([rule1_embedding], lr=0.01)
    optimizer_X4 = torch.optim.Adam([rule2_embedding], lr=0.01)
    optimizer_X5_rule1 = torch.optim.Adam([rule3_embedding], lr=0.01)
    optimizer_X5_rule2 = torch.optim.Adam([rule4_embedding], lr=0.01)
    optimizer_finetune = torch.optim.Adam([rule3_embedding, rule4_embedding], lr=0.01, betas=(0.9, 0.999), eps=1e-05)
    
    print(f"Training started - initial memory usage: {initial_memory:.2f} MB")

    PERFECTION_ACC_THRESHOLD = 0.999 # Define threshold for perfect predicate learning
    perfect_predicates = {'X3': False, 'X4': False, 'X5': False} # Tracks overall predicate perfection

    # Lists to store metrics for plotting (accumulate across cycles)
    all_losses_X3 = []
    all_unobs_accuracies_X3 = []
    all_grad_norms_X3 = []
    all_remaining_samples_X3 = []

    all_losses_X4 = []
    all_unobs_accuracies_X4 = []
    all_grad_norms_X4 = []
    all_remaining_samples_X4 = []

    all_epochs_X5_rule1 = []
    all_losses_X5_rule1 = []
    all_unobs_accuracies_X5_rule1 = []
    all_grad_norms_X5_rule1 = []
    all_remaining_samples_X5_rule1 = []

    all_epochs_X5_rule2 = []
    all_losses_X5_rule2 = []
    all_unobs_accuracies_X5_rule2 = []
    all_grad_norms_X5_rule2 = []
    all_remaining_samples_X5_rule2 = []
    
    all_finetune_losses = []
    all_finetune_accuracies = []
    all_grad_norms_X5_finetune = [] # Added for fine-tune grad norms
    
    all_betas_X5 = [] # Store beta for X5 parts
    all_remaining_samples_X5 = []
    predicates_to_train_in_cycle = ['X3', 'X4', 'X5']

    # Lists to store end-of-cycle accuracies
    all_end_of_cycle_acc_X3 = []
    all_end_of_cycle_acc_X4 = []
    all_end_of_cycle_acc_X5_combined = []

    # Ground truth for structural analysis (optional, analyze_rules can still use it)
    # GROUND_TRUTH_RULES = {
    #     'X3': sorted(('X0', 'X1')),
    #     'X4': sorted(('X2', 'X7')),
    #     'X5_R1': sorted(('X3', 'X6')),
    #     'X5_R2': sorted(('X0', 'X4'))
    # }

    for cycle in range(max_cycles):
        print(f"\n--- Starting Cycle {cycle + 1}/{max_cycles} ---")
        # Print current memory usage at start of each cycle
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Current memory usage: {current_memory:.2f} MB")
        
        # current_train_order = ['X3', 'X4', 'X5']
        current_train_order = random.sample(predicates_to_train_in_cycle, len(predicates_to_train_in_cycle))
        print(f"Cycle {cycle + 1} training order: {current_train_order}")

        for predicate_block_to_train in current_train_order:
            if predicate_block_to_train == 'X3':
                if perfect_predicates['X3']:
                    print(f"  Skipping X3 training in Cycle {cycle+1} as its rule is considered perfect.")
                    continue # Skip to next predicate in current_train_order
                
                print(f"\n-- Cycle {cycle + 1}, Training X3 rule... --")
                remaining_data_X3 = data_set.clone() # Fresh data for this block
                for epoch in range(epochs_per_block):
                    total_loss = 0.0
                    total_unobs_acc = torch.zeros(3) # Index 0 for X3
                    total_grad_norm = torch.zeros(1)
                    n_batches = len(remaining_data_X3) // batch_size
                    if n_batches == 0:
                        print("Warning: Not enough data for a batch in X3 training. Skipping epoch.")
                        all_losses_X3.append(0)
                        all_unobs_accuracies_X3.append(torch.zeros(3))
                        all_grad_norms_X3.append(torch.zeros(1))
                        all_remaining_samples_X3.append(len(remaining_data_X3))
                        continue

                    for i in range(n_batches):
                        batch_data = remaining_data_X3[i * batch_size:(i + 1) * batch_size]
                        if len(batch_data) == 0:
                            continue
                        optimizer_X3.zero_grad()
                        v3, _ = infer_X(batch_data, softmin_temp=args.softmin_temp)
                        X3_obs = batch_data[:, 5]
                        mask_X3 = batch_data[:, 8]
                        true_X3 = batch_data[:, 11]
                        valid_X3 = ~torch.isnan(X3_obs)
                        loss = torch.mean((v3[valid_X3] - X3_obs[valid_X3]) ** 2) if torch.any(valid_X3) else torch.tensor(0.0)
                        if loss.requires_grad:
                            loss.backward()
                            grad_norm = torch.norm(rule1_embedding.grad) if rule1_embedding.grad is not None else torch.tensor(0.0)
                            total_grad_norm += grad_norm
                            optimizer_X3.step()
                            normalize_embeddings()
                        total_loss += loss.item()
                        # Calculate unobserved accuracy for X3
                        unobs_mask_X3 = (mask_X3 == 0)
                        if torch.any(unobs_mask_X3):
                           total_unobs_acc[0] += torch.mean(((v3[unobs_mask_X3] > 0.5).float() == true_X3[unobs_mask_X3].float()).float())
                        else:
                           total_unobs_acc[0] += torch.tensor(0.0) # Handle case with no unobserved samples

                    avg_loss = total_loss / n_batches if n_batches > 0 else 0
                    avg_unobs_acc = total_unobs_acc / n_batches if n_batches > 0 else torch.zeros(3)
                    avg_grad_norm = total_grad_norm / n_batches if n_batches > 0 else torch.zeros(1)
                    
                    all_losses_X3.append(avg_loss)
                    all_unobs_accuracies_X3.append(avg_unobs_acc)
                    all_grad_norms_X3.append(avg_grad_norm.item())
                    all_remaining_samples_X3.append(len(remaining_data_X3))
                    # print(f'  X3 - Cycle {cycle+1}, Epoch {epoch + 1}/{epochs_per_block}, Loss: {avg_loss:.4f}, Unobserved X3 Acc: {avg_unobs_acc[0]:.4f}')
                analyze_rules() # Analyze after X3 block
                cleanup_memory() # Clean up memory after X3 training

            elif predicate_block_to_train == 'X4':
                if perfect_predicates['X4']:
                    print(f"  Skipping X4 training in Cycle {cycle+1} as its rule is considered perfect.")
                    continue

                print(f"\n-- Cycle {cycle + 1}, Training X4 rule... --")
                remaining_data_X4 = data_set.clone() # Fresh data for this block
                for epoch in range(epochs_per_block):
                    total_loss = 0.0
                    total_unobs_acc = torch.zeros(3) # Index 1 for X4
                    total_grad_norm = torch.zeros(1)
                    n_batches = len(remaining_data_X4) // batch_size
                    if n_batches == 0:
                        print("Warning: Not enough data for a batch in X4 training. Skipping epoch.")
                        all_losses_X4.append(0)
                        all_unobs_accuracies_X4.append(torch.zeros(3))
                        all_grad_norms_X4.append(torch.zeros(1))
                        all_remaining_samples_X4.append(len(remaining_data_X4))
                        continue

                    for i in range(n_batches):
                        batch_data = remaining_data_X4[i * batch_size:(i + 1) * batch_size]
                        if len(batch_data) == 0:
                            continue
                        optimizer_X4.zero_grad()
                        _, v4 = infer_X(batch_data, softmin_temp=args.softmin_temp)
                        X4_obs = batch_data[:, 6]
                        mask_X4 = batch_data[:, 9]
                        true_X4 = batch_data[:, 12]
                        valid_X4 = ~torch.isnan(X4_obs)
                        loss = torch.mean((v4[valid_X4] - X4_obs[valid_X4]) ** 2) if torch.any(valid_X4) else torch.tensor(0.0)
                        if loss.requires_grad:
                            loss.backward()
                            grad_norm = torch.norm(rule2_embedding.grad) if rule2_embedding.grad is not None else torch.tensor(0.0)
                            total_grad_norm += grad_norm
                            optimizer_X4.step()
                            normalize_embeddings()
                        total_loss += loss.item()
                        # Calculate unobserved accuracy for X4
                        unobs_mask_X4 = (mask_X4 == 0)
                        if torch.any(unobs_mask_X4):
                            total_unobs_acc[1] += torch.mean(((v4[unobs_mask_X4] > 0.5).float() == true_X4[unobs_mask_X4].float()).float())
                        else:
                            total_unobs_acc[1] += torch.tensor(0.0)

                    avg_loss = total_loss / n_batches if n_batches > 0 else 0
                    avg_unobs_acc = total_unobs_acc / n_batches if n_batches > 0 else torch.zeros(3)
                    avg_grad_norm = total_grad_norm / n_batches if n_batches > 0 else torch.zeros(1)

                    all_losses_X4.append(avg_loss)
                    all_unobs_accuracies_X4.append(avg_unobs_acc)
                    all_grad_norms_X4.append(avg_grad_norm.item())
                    all_remaining_samples_X4.append(len(remaining_data_X4))
                    # print(f'  X4 - Cycle {cycle+1}, Epoch {epoch + 1}/{epochs_per_block}, Loss: {avg_loss:.4f}, Unobserved X4 Acc: {avg_unobs_acc[1]:.4f}')
                analyze_rules() # Analyze after X4 block
                cleanup_memory() # Clean up memory after X4 training

            elif predicate_block_to_train == 'X5':
                if perfect_predicates['X5']:
                    print(f"  Skipping X5 training (Rule 1, Rule 2, Fine-tune) in Cycle {cycle+1} as its rule is considered perfect.")
                    continue

                print(f"\n-- Cycle {cycle + 1}, Training X5 rules (Rule 1, Rule 2, Fine-tune)... --")
                # X5 Rule 1 Learning (with hard covering)
                print(f"\n  -- Cycle {cycle + 1}, Learning X5 Rule 1 --")
                remaining_data_X5_R1 = data_set.clone() # Start with full dataset for rule 1
                if args.beta_mode == 'constant':
                    beta_schedule = np.full(epochs_per_block, args.beta_const)
                else:
                    beta_schedule = frange_cycle_cosine_decay(
                        epochs_per_block,
                        start=args.beta_start,
                        stop=args.beta_stop,
                        n_cycle=args.beta_n_cycle,
                        ratio=args.beta_ratio,
                        decay=args.beta_decay,
                    )

                for epoch in range(epochs_per_block):
                    beta = beta_schedule[epoch]
                    all_betas_X5.append(beta)
                    # print(f"Epoch {epoch + 1}, Beta: {beta:.4f}")
                    total_loss = 0.0
                    total_unobs_acc = torch.zeros(3) # Index 2 for X5
                    total_grad_norm = torch.zeros(1)
                    n_batches = len(remaining_data_X5_R1) // batch_size
                    if n_batches == 0:
                        print("Warning: Not enough data for a batch in X5 Rule 1 training. Skipping epoch.")
                        all_losses_X5_rule1.append(0)
                        all_unobs_accuracies_X5_rule1.append(torch.zeros(3))
                        all_grad_norms_X5_rule1.append(torch.zeros(1))
                        all_remaining_samples_X5_rule1.append(len(remaining_data_X5_R1))
                        all_remaining_samples_X5.append(len(remaining_data_X5_R1))
                        continue
                        
                    for i in range(n_batches):
                        batch_data = remaining_data_X5_R1[i * batch_size:(i + 1) * batch_size]
                        if len(batch_data) == 0:
                            continue
                        optimizer_X5_rule1.zero_grad()
                        v5 = infer_v5(batch_data, beta, softmin_temp=args.softmin_temp, rule_idx=0)
                        X5_obs = batch_data[:, 7]
                        mask_X5 = batch_data[:, 10]
                        true_X5 = batch_data[:, 13]
                        valid_X5 = ~torch.isnan(X5_obs)
                        loss = torch.mean((v5[valid_X5] - X5_obs[valid_X5]) ** 2) if torch.any(valid_X5) else torch.tensor(0.0)
                        if loss.requires_grad:
                            loss.backward()
                            grad_norm = torch.norm(rule3_embedding.grad) if rule3_embedding.grad is not None else torch.tensor(0.0)
                            total_grad_norm += grad_norm
                            optimizer_X5_rule1.step()
                            normalize_embeddings()
                            with torch.no_grad():
                                rule3_embedding.data = torch.relu(rule3_embedding.data)
                        total_loss += loss.item()
                        unobs_mask_X5 = (mask_X5 == 0)
                        if torch.any(unobs_mask_X5):
                            total_unobs_acc[2] += torch.mean(((v5[unobs_mask_X5] > 0.5).float() == true_X5[unobs_mask_X5].float()).float())
                        else:
                            total_unobs_acc[2] += torch.tensor(0.0)


                    avg_loss = total_loss / n_batches if n_batches > 0 else 0
                    avg_unobs_acc = total_unobs_acc / n_batches if n_batches > 0 else torch.zeros(3)
                    avg_grad_norm = total_grad_norm / n_batches if n_batches > 0 else torch.zeros(1)
                    
                    all_epochs_X5_rule1.append(epoch)
                    all_losses_X5_rule1.append(avg_loss)
                    all_unobs_accuracies_X5_rule1.append(avg_unobs_acc)
                    all_grad_norms_X5_rule1.append(avg_grad_norm.item())
                    all_remaining_samples_X5_rule1.append(len(remaining_data_X5_R1))
                    all_remaining_samples_X5.append(len(remaining_data_X5_R1))
                    # print(f'    X5 Rule 1 - Cycle {cycle+1}, Epoch {epoch + 1}/{epochs_per_block}, Loss: {avg_loss:.4f}, Unobserved X5 Acc: {avg_unobs_acc[2]:.4f}')
                    
                    # Hard covering for X5 Rule 1
                    if epoch > 0 : #and epoch % 1 == 0:
                        mask = remaining_data_X5_R1[:, 13] == 1
                        if torch.any(mask):
                            v5_cover = infer_v5(remaining_data_X5_R1, beta, softmin_temp=args.softmin_temp, rule_idx=0) # Use current remaining data
                            well_explained = (v5_cover > 0.99)
                            to_remove_indices = well_explained.nonzero(as_tuple=True)[0]
                            
                            # Create a mask for removal from remaining_data_X5_R1
                            remove_mask_for_remaining = torch.zeros(len(remaining_data_X5_R1), dtype=torch.bool)
                            remove_mask_for_remaining[to_remove_indices] = True
                            remaining_data_X5_R1 = remaining_data_X5_R1[~remove_mask_for_remaining]

                        num_well_explained = well_explained.sum().item() if 'well_explained' in locals() else 0
                        # print(f"    X5 Rule 1 - Remaining samples after covering: {len(remaining_data_X5_R1)}, Well-explained: {num_well_explained}")
                        if num_well_explained > 0: # Heuristic to move on
                            print(f"    X5 Rule 1 - Moving to Rule 2 due to significant covering.")
                            all_remaining_samples_X5.append(len(remaining_data_X5_R1))
                            break

                # X5 Rule 2 Learning (with hard covering, on data NOT covered by Rule 1)
                print(f"\n  -- Cycle {cycle + 1}, Learning X5 Rule 2 --")
                # remaining_data_X5_R2 starts from where R1 left off (after its covering)
                remaining_data_X5_R2 = remaining_data_X5_R1.clone() 
                if args.beta_mode == 'constant':
                    beta_schedule = np.full(epochs_per_block, args.beta_const)
                else:
                    beta_schedule = frange_cycle_cosine_decay(
                        epochs_per_block,
                        start=args.beta_start,
                        stop=args.beta_stop,
                        n_cycle=args.beta_n_cycle,
                        ratio=args.beta_ratio,
                        decay=args.beta_decay,
                    )

                for epoch in range(epochs_per_block):
                    beta = beta_schedule[epoch]
                    all_betas_X5.append(beta)
                    # print(f"Epoch {epoch + 1}, Beta: {beta:.4f}")
                    total_loss = 0.0
                    total_unobs_acc = torch.zeros(3) # Index 2 for X5
                    total_grad_norm = torch.zeros(1)
                    n_batches = len(remaining_data_X5_R2) // batch_size
                    if n_batches == 0:
                        print("Warning: Not enough data for a batch in X5 Rule 2 training. Skipping epoch.")
                        all_losses_X5_rule2.append(0)
                        all_unobs_accuracies_X5_rule2.append(torch.zeros(3))
                        all_grad_norms_X5_rule2.append(torch.zeros(1))
                        all_remaining_samples_X5_rule2.append(len(remaining_data_X5_R2))
                        all_remaining_samples_X5.append(len(remaining_data_X5_R2))
                        continue

                    for i in range(n_batches):
                        batch_data = remaining_data_X5_R2[i * batch_size:(i + 1) * batch_size]
                        if len(batch_data) == 0:
                            continue
                        optimizer_X5_rule2.zero_grad()
                        v5 = infer_v5(batch_data, beta, softmin_temp=args.softmin_temp, rule_idx=1)
                        X5_obs = batch_data[:, 7]
                        mask_X5 = batch_data[:, 10]
                        true_X5 = batch_data[:, 13]
                        valid_X5 = ~torch.isnan(X5_obs)
                        loss = torch.mean((v5[valid_X5] - X5_obs[valid_X5]) ** 2) if torch.any(valid_X5) else torch.tensor(0.0)
                        if loss.requires_grad:
                            loss.backward()
                            grad_norm = torch.norm(rule4_embedding.grad) if rule4_embedding.grad is not None else torch.tensor(0.0)
                            total_grad_norm += grad_norm
                            optimizer_X5_rule2.step()
                            normalize_embeddings()
                            with torch.no_grad():
                                rule4_embedding.data = torch.relu(rule4_embedding.data)
                        total_loss += loss.item()
                        unobs_mask_X5 = (mask_X5 == 0)
                        if torch.any(unobs_mask_X5):
                            total_unobs_acc[2] += torch.mean(((v5[unobs_mask_X5] > 0.5).float() == true_X5[unobs_mask_X5].float()).float())
                        else:
                            total_unobs_acc[2] += torch.tensor(0.0)

                    avg_loss = total_loss / n_batches if n_batches > 0 else 0
                    avg_unobs_acc = total_unobs_acc / n_batches if n_batches > 0 else torch.zeros(3)
                    avg_grad_norm = total_grad_norm / n_batches if n_batches > 0 else torch.zeros(1)
                    
                    all_epochs_X5_rule2.append(epoch)
                    all_losses_X5_rule2.append(avg_loss)
                    all_unobs_accuracies_X5_rule2.append(avg_unobs_acc)
                    all_grad_norms_X5_rule2.append(avg_grad_norm.item())
                    all_remaining_samples_X5_rule2.append(len(remaining_data_X5_R2))
                    all_remaining_samples_X5.append(len(remaining_data_X5_R2))
                    # print(f'    X5 Rule 2 - Cycle {cycle+1}, Epoch {epoch + 1}/{epochs_per_block}, Loss: {avg_loss:.4f}, Unobserved X5 Acc: {avg_unobs_acc[2]:.4f}')

                    # Hard covering for X5 Rule 2
                    if epoch > 0 : #and epoch % 1 == 0:
                        mask = remaining_data_X5_R2[:, 13] == 1 # Check against true X5
                        if torch.any(mask):
                            v5_cover = infer_v5(remaining_data_X5_R2, beta, softmin_temp=args.softmin_temp, rule_idx=1) # Use current remaining data for rule 2
                            well_explained = (v5_cover > 0.99)
                            to_remove_indices = well_explained.nonzero(as_tuple=True)[0]

                            remove_mask_for_remaining_r2 = torch.zeros(len(remaining_data_X5_R2), dtype=torch.bool)
                            remove_mask_for_remaining_r2[to_remove_indices] = True
                            remaining_data_X5_R2 = remaining_data_X5_R2[~remove_mask_for_remaining_r2]
                            
                        num_well_explained = well_explained.sum().item() if 'well_explained' in locals() else 0
                        # print(f"    X5 Rule 2 - Remaining samples after covering: {len(remaining_data_X5_R2)}, Well-explained: {num_well_explained}")
                        if num_well_explained > 0: # Heuristic
                            print(f"    X5 Rule 2 - Moving to Fine-tune due to significant covering.")
                            all_remaining_samples_X5.append(len(remaining_data_X5_R2))
                            break
                analyze_rules()

                # Fine-tuning X5 rules jointly
                print(f"\n  -- Cycle {cycle + 1}, Fine-tuning both X5 rules jointly --")
                remaining_data_finetune = data_set.clone() # Use full dataset for fine-tuning
                if args.beta_mode == 'constant':
                    beta_schedule = np.full(epochs_per_block, args.beta_const)
                else:
                    beta_schedule = frange_cycle_cosine_decay(
                        epochs_per_block,
                        start=args.beta_start,
                        stop=args.beta_stop,
                        n_cycle=args.beta_n_cycle,
                        ratio=args.beta_ratio,
                        decay=args.beta_decay,
                    )

                for epoch in range(epochs_per_block):
                    beta = beta_schedule[epoch]
                    all_betas_X5.append(beta) # Also track betas during fine-tune
                    total_loss = 0.0
                    total_grad_norm_ft = 0.0 # For fine-tune grad norm
                    n_batches = len(remaining_data_finetune) // batch_size
                    if n_batches == 0:
                        print("Warning: Not enough data for a batch in X5 Fine-tune. Skipping epoch.")
                        all_finetune_losses.append(0)
                        all_finetune_accuracies.append(0) # Assuming accuracy returns a single float
                        continue

                    for i in range(n_batches):
                        batch_data = remaining_data_finetune[i * batch_size:(i + 1) * batch_size]
                        if len(batch_data) == 0:
                            continue
                        optimizer_finetune.zero_grad()
                        v5 = infer_v5(batch_data, beta, softmin_temp=args.softmin_temp)  # Combined output (soft-OR)
                        X5_obs = batch_data[:, 7]
                        valid_X5 = ~torch.isnan(X5_obs)
                        mse_loss = torch.mean((v5[valid_X5] - X5_obs[valid_X5]) ** 2) if torch.any(valid_X5) else torch.tensor(0.0)
                        diversity_loss = -torch.abs(torch.dot(rule3_embedding.flatten(), rule4_embedding.flatten()))
                        loss = mse_loss + 0.00 * diversity_loss
                        if loss.requires_grad:
                            loss.backward()
                            optimizer_finetune.step()
                            normalize_embeddings()
                            with torch.no_grad():
                                rule3_embedding.data = torch.relu(rule3_embedding.data)
                                rule4_embedding.data = torch.relu(rule4_embedding.data)
                        total_loss += loss.item()
                        
                        # Calculate and accumulate fine-tune gradient norm
                        current_grad_norm_ft = 0.0
                        if rule3_embedding.grad is not None:
                            current_grad_norm_ft += torch.norm(rule3_embedding.grad).item()
                        if rule4_embedding.grad is not None:
                            current_grad_norm_ft += torch.norm(rule4_embedding.grad).item() # Summing norms
                        total_grad_norm_ft += current_grad_norm_ft

                    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
                    avg_grad_norm_ft = total_grad_norm_ft / n_batches if n_batches > 0 else 0.0
                    # Calculate overall accuracy for X5 using the combined model after fine-tuning step
                    acc_X3_unobs, acc_X4_unobs, acc_X5_unobs_combined = calculate_unobserved_accuracy(remaining_data_finetune, beta, softmin_temp=args.softmin_temp)

                    all_finetune_losses.append(avg_loss)
                    all_finetune_accuracies.append(acc_X5_unobs_combined) # Store X5 specific unobserved accuracy
                    all_grad_norms_X5_finetune.append(avg_grad_norm_ft) # Store fine-tune grad norm
                    if epoch < 28:
                        all_remaining_samples_X5.append(len(remaining_data_finetune))
                    # print(f'    X5 Fine-tune - Cycle {cycle+1}, Epoch {epoch + 1}/{epochs_per_block}, Loss: {avg_loss:.4f}, Unobserved X5 Acc (Combined): {acc_X5_unobs_combined:.4f}')
                    if acc_X5_unobs_combined > 0.99 : # Heuristic based on X5 combined performance
                        pass # Keep training fine-tune for full epochs_per_block unless other stopping condition
                analyze_rules() # Final analysis after X5 block
                cleanup_memory() # Clean up memory after X5 training
        
        # --- End of a Cycle Evaluation --- 
        print(f"\n--- Cycle {cycle + 1} Completed. Evaluating predicate perfection. ---")
        eval_beta = all_betas_X5[-1] if all_betas_X5 else 10.0 # Use last beta or a default for consistent evaluation
        current_unobs_acc_X3, current_unobs_acc_X4, current_unobs_acc_X5_combined = calculate_unobserved_accuracy(data_set, eval_beta, softmin_temp=args.softmin_temp)
    
        all_end_of_cycle_acc_X3.append(current_unobs_acc_X3)
        all_end_of_cycle_acc_X4.append(current_unobs_acc_X4)
        all_end_of_cycle_acc_X5_combined.append(current_unobs_acc_X5_combined)

        if not perfect_predicates['X3'] and current_unobs_acc_X3 >= PERFECTION_ACC_THRESHOLD:
            perfect_predicates['X3'] = True
            # print(f"    ✓ Predicate X3 is now considered perfect (Unobs Acc: {current_unobs_acc_X3:.4f}).")
        
        if not perfect_predicates['X4'] and current_unobs_acc_X4 >= PERFECTION_ACC_THRESHOLD:
            perfect_predicates['X4'] = True
            # print(f"    ✓ Predicate X4 is now considered perfect (Unobs Acc: {current_unobs_acc_X4:.4f}).")

        if not perfect_predicates['X5'] and current_unobs_acc_X5_combined >= PERFECTION_ACC_THRESHOLD:
            perfect_predicates['X5'] = True
            # print(f"    ✓ Predicate X5 (Combined) is now considered perfect (Unobs Acc: {current_unobs_acc_X5_combined:.4f}).")

        print(f"  Current Perfect Predicates Status: {perfect_predicates}")
        analyze_rules() # Show current structural rules at end of cycle
        
        if perfect_predicates['X3'] and perfect_predicates['X4'] and perfect_predicates['X5']:
            print(f"All predicates (X3, X4, X5) have achieved perfect inference accuracy in Cycle {cycle + 1}. Stopping training.")
            break 
        if cycle == max_cycles - 1:
            print("Reached maximum number of cycles.")

    # Final analysis after all cycles
    print("\n--- All Training Cycles Completed ---")
    analyze_rules()
    print(f"  End-of-Cycle Unobserved Accuracies: X3: {current_unobs_acc_X3:.4f}, X4: {current_unobs_acc_X4:.4f}, X5 (Combined): {current_unobs_acc_X5_combined:.4f}")

    
    # --- PLOTTING ---
    # Consolidate losses for X5 for unified plotting if desired, or plot separately
    
    plt.figure(figsize=(18, 12)) # Adjusted for 2x2 grid
    plt.rcParams.update({
        'font.size': 22, 
        'axes.titlesize': 22, 
        'axes.labelsize': 22, 
        'xtick.labelsize': 22, 
        'ytick.labelsize': 22, 
        'legend.fontsize': 22,
        'lines.markersize': 6,
        'lines.linewidth': 3
    })

    # --- Subplot 1: Losses (Top-Left) ---
    plt.subplot(2, 2, 1)
    if all_losses_X3: 
        plt.plot(range(1, len(all_losses_X3) + 1), all_losses_X3, marker='o', label='X3 Loss')
        for i in range(1, int(len(all_losses_X3) / args.epochs_per_block) + 1):
            plt.axvline(x=i * args.epochs_per_block, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    if all_losses_X4: 
        plt.plot(range(1, len(all_losses_X4) + 1), all_losses_X4, marker='o', label='X4 Loss')
        for i in range(1, int(len(all_losses_X4) / args.epochs_per_block) + 1):
            plt.axvline(x=i * args.epochs_per_block, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    
    x5_combined_losses = all_losses_X5_rule1 + all_losses_X5_rule2 + all_finetune_losses
    if x5_combined_losses:
        plt.plot(range(1, len(x5_combined_losses) + 1), x5_combined_losses, marker='o', label='X5 Loss')
        current_offset = 0
        for segment_data in [all_losses_X5_rule1, all_losses_X5_rule2, all_finetune_losses]:
            if segment_data: # Check if the segment has data
                for i in range(1, int(len(segment_data) / args.epochs_per_block) + 1):
                    plt.axvline(x=current_offset + i * args.epochs_per_block, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
            current_offset += len(segment_data)
    
    plt.title(f'(a) Predicate Rule Training Losses (Seed {args.seed})')
    plt.xlabel('Predicate Training Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- Subplot 2: Unobserved Accuracies (Top-Right) ---
    plt.subplot(2, 2, 2)
    if all_unobs_accuracies_X3:
        unobs_acc_X3_plot = [item[0].item() if torch.is_tensor(item[0]) else item[0] for item in all_unobs_accuracies_X3]
        plt.plot(range(1, len(unobs_acc_X3_plot) + 1), unobs_acc_X3_plot, marker='o', label='X3 Unobserved Acc.')
        for i in range(1, int(len(unobs_acc_X3_plot) / args.epochs_per_block) + 1):
            plt.axvline(x=i * args.epochs_per_block, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    if all_unobs_accuracies_X4:
        unobs_acc_X4_plot = [item[1].item() if torch.is_tensor(item[1]) else item[1] for item in all_unobs_accuracies_X4]
        plt.plot(range(1, len(unobs_acc_X4_plot) + 1), unobs_acc_X4_plot, marker='o', label='X4 Unobserved Acc.')
        for i in range(1, int(len(unobs_acc_X4_plot) / args.epochs_per_block) + 1):
            plt.axvline(x=i * args.epochs_per_block, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    
    x5_r1_acc = [item[2].item() if torch.is_tensor(item[2]) else item[2] for item in all_unobs_accuracies_X5_rule1]
    x5_r2_acc = [item[2].item() if torch.is_tensor(item[2]) else item[2] for item in all_unobs_accuracies_X5_rule2]
    # all_finetune_accuracies is already a list of floats
    x5_combined_unobs_acc = x5_r1_acc + x5_r2_acc + all_finetune_accuracies 
    if x5_combined_unobs_acc:
        plt.plot(range(1, len(x5_combined_unobs_acc) + 1), x5_combined_unobs_acc, marker='o', label='X5 Unobserved Acc.')
        current_offset = 0
        # Note: all_finetune_accuracies is directly used, x5_r1_acc and x5_r2_acc are processed lists
        for segment_data in [x5_r1_acc, x5_r2_acc, all_finetune_accuracies]:
            if segment_data: # Check if the segment has data
                for i in range(1, int(len(segment_data) / args.epochs_per_block) + 1):
                    plt.axvline(x=current_offset + i * args.epochs_per_block, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
            current_offset += len(segment_data)

    plt.title(f'(b) Unobserved Predicate Imputation Accuracies (Seed {args.seed})')
    plt.xlabel('Predicate Training Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # --- Subplot 3: Gradient Norms (Bottom-Left) ---
    plt.subplot(2, 2, 3)
    if all_grad_norms_X3: 
        plt.plot(range(1, len(all_grad_norms_X3) + 1), all_grad_norms_X3, marker='o', label='X3 Grad Norm')
        for i in range(1, int(len(all_grad_norms_X3) / args.epochs_per_block) + 1):
            plt.axvline(x=i * args.epochs_per_block, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    if all_grad_norms_X4: 
        plt.plot(range(1, len(all_grad_norms_X4) + 1), all_grad_norms_X4, marker='o', label='X4 Grad Norm')
        for i in range(1, int(len(all_grad_norms_X4) / args.epochs_per_block) + 1):
            plt.axvline(x=i * args.epochs_per_block, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    
    x5_combined_grad_norms = all_grad_norms_X5_rule1 + all_grad_norms_X5_rule2 + all_grad_norms_X5_finetune
    if x5_combined_grad_norms:
        plt.plot(range(1, len(x5_combined_grad_norms) + 1), x5_combined_grad_norms, marker='o', label='X5 Grad Norm')
        current_offset = 0
        for segment_data in [all_grad_norms_X5_rule1, all_grad_norms_X5_rule2, all_grad_norms_X5_finetune]:
            if segment_data: # Check if the segment has data
                for i in range(1, int(len(segment_data) / args.epochs_per_block) + 1):
                    plt.axvline(x=current_offset + i * args.epochs_per_block, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
            current_offset += len(segment_data)

    plt.title(f'(c) Predicate Rule Gradient Norms (Seed {args.seed})')
    plt.xlabel('Predicate Training Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)

    # --- Subplot 4: Overall Unobserved Accuracy per Cycle (Bottom-Right) ---
    plt.subplot(2, 2, 4)
    cycles = range(1, len(all_end_of_cycle_acc_X3) + 1) # Assumes all lists have same length from previous logic

    if all_end_of_cycle_acc_X3:
        plt.plot(cycles, all_end_of_cycle_acc_X3, marker='o', markersize=6, linestyle='-', label='X3 End-of-Cycle Acc.')
    if all_end_of_cycle_acc_X4:
        plt.plot(cycles, all_end_of_cycle_acc_X4, marker='s', markersize=6, linestyle='-', label='X4 End-of-Cycle Acc.')
    if all_end_of_cycle_acc_X5_combined:
        plt.plot(cycles, all_end_of_cycle_acc_X5_combined, marker='^', markersize=6, linestyle='-', label='X5 End-of-Cycle Acc.')

    plt.title(f'Overall Unobserved Accuracy per Cycle (Seed {args.seed})')
    plt.xlabel('Cycle Number')
    plt.ylabel('Unobserved Accuracy')
    if cycles: # Check if there are any cycles to plot
        # Ensure integer ticks for cycle numbers, and that x-axis starts at 1 if there are cycles.
        actual_xticks = list(cycles)
        if not actual_xticks: # Handle case of no cycles if necessary, though unlikely if plotting
            actual_xticks = [1] 
        plt.xticks(actual_xticks) 
        plt.xlim(left=max(0.5, min(actual_xticks) - 0.5), right=max(actual_xticks) + 0.5) # Adjust x-limits for better view
        
    if all_end_of_cycle_acc_X3 or all_end_of_cycle_acc_X4 or all_end_of_cycle_acc_X5_combined:
        plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05) # Typical accuracy range

    # Final calls for the entire figure
    plt.tight_layout(pad=3.0) 
    plt.savefig(os.path.join(args.output_dir, 'plots', f'rule_b_cycles_seed_{args.seed}_obs{args.obs_prob}_n{args.n_samples}.pdf'))
    plt.close()

    print(f"All plots saved to {os.path.join(args.output_dir, 'plots')} with seed {args.seed}")
    
    # Calculate and print final timing and memory usage
    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Get peak memory (platform dependent)
    try:
        peak_memory = process.memory_info().peak_rss / 1024 / 1024  # MB (Linux/Mac)
    except AttributeError:
        try:
            peak_memory = process.memory_info().peak_wset / 1024 / 1024  # MB (Windows)
        except AttributeError:
            peak_memory = final_memory  # Fallback if peak memory not available
    
    total_time = end_time - start_time
    memory_used = final_memory - initial_memory
    
    # Save timing and memory info to file
    # Build a beta tag and softmin tag for unique file naming
    if args.beta_mode == "constant":
        beta_tag = f"const{args.beta_const}"
    else:
        beta_tag = f"cycle{args.beta_start}-{args.beta_stop}"
    softmin_tag = f"smt{args.softmin_temp}"

    timing_info = {
        "seed": args.seed,
        "obs_prob": args.obs_prob,
        "n_samples": args.n_samples,
        "epochs_per_block": args.epochs_per_block,
        "max_cycles": args.max_cycles,
        "beta_mode": args.beta_mode,
        "beta_const": args.beta_const,
        "beta_start": args.beta_start,
        "beta_stop": args.beta_stop,
        "softmin_temp": args.softmin_temp,
        "final_unobs_acc_X3": current_unobs_acc_X3,
        "final_unobs_acc_X4": current_unobs_acc_X4,
        "final_unobs_acc_X5_combined": current_unobs_acc_X5_combined,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time/60,
        "total_time_hours": total_time/3600 if total_time > 3600 else None,
        "initial_memory_mb": initial_memory,
        "final_memory_mb": final_memory,
        "memory_increase_mb": memory_used,
        "peak_memory_mb": peak_memory,
        "memory_efficiency_mb_per_sec": memory_used/total_time if total_time > 0 else 0
    }
    
    timing_file = os.path.join(
        args.output_dir,
        f'timing_info_seed_{args.seed}_obs{args.obs_prob}_n{args.n_samples}_beta_{beta_tag}_{softmin_tag}.json'
    )
    with open(timing_file, 'w') as f:
        json.dump(timing_info, f, indent=4)
    

# Generate data and train model
if __name__ == "__main__":
    print(f"Running with random seed: {args.seed}, obs_prob: {args.obs_prob}, n_samples: {args.n_samples}, epochs_per_block: {args.epochs_per_block}, max_cycles: {args.max_cycles}")
    print(f"Missing mechanism: {args.missing_mechanism}")
    if args.missing_mechanism == 'MAR':
        print(f"MAR dependency variable: {args.mar_dependency}")
    elif args.missing_mechanism == 'MNAR':
        print(f"MNAR threshold: {args.mnar_threshold}")
    
    data_set = generate_data(
        n_samples=args.n_samples, 
        p_X0=0.5, p_X1=0.5, p_X2=0.5, p_X6=0.5, p_X7=0.5, 
        obs_prob=args.obs_prob,
        missing_mechanism=args.missing_mechanism,
        mar_dependency=args.mar_dependency,
        mnar_threshold=args.mnar_threshold
    )
    train(data_set, batch_size=64, epochs_per_block=args.epochs_per_block, max_cycles=args.max_cycles)