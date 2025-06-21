import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
from typing import Optional, Tuple
import sys
import argparse
import csv

# --- Import necessary components from transformers internals ---
# This is crucial for our custom LlamaAttention class to work.
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb


# Suppress Hugging Face warnings about sequence length
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using `max_length`*")

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
# MODEL_NAME = "EleutherAI/gpt-neo-125m"
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = './IMDB_Dataset_100.csv'
NOISE_SEED = 42
NOISE_LEVEL = 1
NUM_SHOTS = 4

# A dictionary to pass configuration to the custom layer
# This will be attached directly to the custom module.
perturb_config = {"head_to_perturb": None, "noise_level": NOISE_LEVEL}


# --- 2. DATASET PREPARATION ---
print("--- Step 1: Loading and Preparing Dataset ---")

# Load the local CSV file
try:
    full_dataset = load_dataset('csv', data_files=DATASET_PATH)['train']
except FileNotFoundError:
    print(f"Error: The dataset file '{DATASET_PATH}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script.")
    exit()

## Split the data
np.random.seed(42)
# Split by sentiment
positive_samples = [i for i, item in enumerate(full_dataset) if item['sentiment'] == 'positive']
negative_samples = [i for i, item in enumerate(full_dataset) if item['sentiment'] == 'negative']
# Randomly select NUM_SHOTS/2 samples from each sentiment
shot_indices = (
    np.random.choice(positive_samples, NUM_SHOTS//2, replace=False).tolist() +
    np.random.choice(negative_samples, NUM_SHOTS//2, replace=False).tolist()
)
# Get remaining indices for test set
test_indices = [i for i in range(len(full_dataset)) if i not in shot_indices]
# Create the datasets
shots_dataset = full_dataset.select(shot_indices)
test_dataset = full_dataset.select(test_indices)

# Construct the few-shot ICL prompt context
context_prompt = ""
for i in range(NUM_SHOTS):
    review = shots_dataset[i]['review']
    sentiment = shots_dataset[i]['sentiment']
    context_prompt += f"Review: \"{review}\"\nSentiment: {sentiment}\n\n"

print(f"Loaded {len(test_dataset)} samples for evaluation.")
print("Few-shot context prompt created successfully.\n")


# --- 3. LOAD MODEL AND TOKENIZER ---
print("--- Step 2: Loading Model and Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.config.pad_token_id = tokenizer.pad_token_id

print(f"Model '{MODEL_NAME}' loaded to {DEVICE}.\n")


# --- 4. CUSTOM PERTURBATION LAYER DEFINITION ---

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class PerturbedLlamaAttention(LlamaAttention):
    """
    A custom LlamaAttention layer that injects noise into a specific attention
    head's output before the final projection layer.
    """
    def __init__(self, config, layer_idx=None):
        # Initialize the parent LlamaAttention class
        super().__init__(config, layer_idx)
        # Add our custom config holder
        self.perturb_config = {"head_to_perturb": None, "noise_level_fraction_of_max": 0.5, "noise_seed": 42}

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        
        # --- !!! START OF OUR CUSTOM PERTURBATION LOGIC !!! ---
        head_idx_to_perturb = self.perturb_config.get("head_to_perturb")
        if head_idx_to_perturb is not None:
            p = self.perturb_config.get("noise_level_fraction_of_max", 0.5)
            noise_seed = self.perturb_config.get("noise_seed", 42)
            
            # Select the specific head's output to perturb
            head_output = attn_output[:, :, head_idx_to_perturb, :]
            
            # Compute dynamic noise level as p% of absolute max value in head_output
            max_val = head_output.abs().max()
            noise_level = p * max_val.item()
            
            # Generate noise with the same shape and device as the head's output
            torch.manual_seed(noise_seed)
            noise = torch.randn_like(head_output) * noise_level
            
            # Add the noise to the specified head
            attn_output[:, :, head_idx_to_perturb, :] += noise
        # --- !!! END OF OUR CUSTOM PERTURBATION LOGIC !!! ---

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# --- 5. COMPREHENSIVE EVALUATION FUNCTION ---
def evaluate_model(model, tokenizer, dataset, context):
    model.eval()
    all_predictions = []
    all_true_labels = []

    for item in tqdm(dataset, desc="Evaluating", leave=False, disable=False):
        review_text = item['review']
        true_label_str = item['sentiment']
        full_prompt = context + f"Review: \"{review_text}\"\nSentiment:"
        
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt",
            max_length=4096,
            padding=True,
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=3,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1
            )
            prediction_text = tokenizer.decode(output_sequences[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
            
            if "positive" in prediction_text:
                predicted_label = "positive"
            elif "negative" in prediction_text:
                predicted_label = "negative"
            else: # Fallback for unexpected generation
                predicted_label = "positive" if true_label_str == "negative" else "negative"

            all_predictions.append(predicted_label)
            all_true_labels.append(true_label_str)
                
    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    
    return {"accuracy": accuracy, "f1_score": f1}


# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run attention head perturbation analysis.")
    parser.add_argument("--noise_level_fraction_of_max", type=float, default=0.50, help="Fraction of max value for noise level.")
    parser.add_argument("--noise_seed", type=int, default=42, help="Seed for noise generation.")
    args = parser.parse_args()

    # Update global variables with parsed arguments
    NOISE_LEVEL = args.noise_level_fraction_of_max
    NOISE_SEED = args.noise_seed

    print("--- Step 3: Running Baseline Evaluation ---")
    result_original = evaluate_model(model, tokenizer, test_dataset, context_prompt)
    print(f"Baseline Metrics -> Accuracy: {result_original['accuracy']:.4f}, F1-Score: {result_original['f1_score']:.4f}\n")

    print("--- Step 4 & 5: Perturbing Heads and Calculating Contribution ---")

    # Get a reference to the original last attention layer
    original_last_layer_attention = model.model.layers[-1].self_attn
    num_attention_heads = original_last_layer_attention.config.num_attention_heads

    head_contributions = []

    for i in range(num_attention_heads):
        print(f"Perturbing head {i}/{num_attention_heads-1}...")
        
        # --- Create and swap in the custom layer ---
        # 1. Create an instance of our perturbed attention layer
        perturbed_layer = PerturbedLlamaAttention(
            original_last_layer_attention.config,
            layer_idx=len(model.model.layers)-1 # layer_idx is optional but good practice
        ).to(DEVICE)

        # 2. Copy the trained weights from the original layer
        perturbed_layer.load_state_dict(original_last_layer_attention.state_dict())
        
        # 3. Set the perturbation config for the current head
        perturbed_layer.perturb_config["head_to_perturb"] = i
        perturbed_layer.perturb_config["noise_level_fraction_of_max"] = NOISE_LEVEL
        perturbed_layer.perturb_config["noise_seed"] = NOISE_SEED
        
        # 4. Replace the original layer with our custom one
        model.model.layers[-1].self_attn = perturbed_layer

        # Evaluate the model with the perturbation active
        perturbed_metrics = evaluate_model(model, tokenizer, test_dataset, context_prompt)
        
        # IMPORTANT: Restore the original layer to reset the model for the next iteration
        model.model.layers[-1].self_attn = original_last_layer_attention

        # Calculate contribution (performance degradation)
        contribution_accuracy = result_original['accuracy'] - perturbed_metrics['accuracy']
        contribution_f1 = result_original['f1_score'] - perturbed_metrics['f1_score']
        
        head_contributions.append({
            "head": i,
            "contribution_accuracy": contribution_accuracy,
            "contribution_f1": contribution_f1,
        })

    # --- Step 6: Rank Attention Heads ---
    ranked_heads = sorted(head_contributions, key=lambda x: x['contribution_accuracy'], reverse=True)
    
    print("\n--- Final Ranking of Attention Heads (by Operational Contribution) ---")
    print(f"Baseline -> Accuracy: {result_original['accuracy']:.4f}, F1-Score: {result_original['f1_score']:.4f}")
    print("-" * 65)
    print(f"{'Rank':<5} | {'Head':<5} | {'F1 Drop (Contribution)':<25} | {'Accuracy Drop':<15}")
    print("-" * 65)
    for rank, head_info in enumerate(ranked_heads, 1):
        print(f"{rank:<5} | {head_info['head']:<5} | {head_info['contribution_f1']:.4f}{' ':<21}| {head_info['contribution_accuracy']:.4f}")
    print("-" * 65)

    # --- Write Results to CSV File ---
    # Parameterize the file name based on passed arguments
    csv_file_name = f"./results/results_noise_{NOISE_LEVEL}_seed_{NOISE_SEED}.csv"

    # Write results to the CSV file
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Head", "F1 Drop (Contribution)", "Accuracy Drop"])
        for rank, head_info in enumerate(ranked_heads, 1):
            writer.writerow([rank, head_info['head'], head_info['contribution_f1'], head_info['contribution_accuracy']])

    print(f"\nResults written to {csv_file_name}")

