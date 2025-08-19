#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sqlite3
import sys
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import spacy
import torch
from captum.attr import IntegratedGradients, Saliency
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ---
# SECTION 1: DATA HELPER FUNCTIONS
# ---

def get_schema_from_db(db_path):
    """Extracts only the CREATE TABLE statements from a SQLite database."""
    if not os.path.exists(db_path):
        print(f"Warning: Database not found at {db_path}", file=sys.stderr)
        return ""
    try:
        # Connect in read-only mode
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schema_sql = [row[0] for row in cursor.fetchall()]
        conn.close()
        return "\n".join(schema_sql)
    except sqlite3.Error as e:
        print(f"Error reading schema from {db_path}: {e}", file=sys.stderr)
        return ""

def load_evaluation_data(path):
    """Loads evaluation data from a standard multi-line JSON file."""
    print(f"Attempting to load standard JSON from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            # Read the entire file content and parse it as a single JSON object
            data = json.load(f)
            # The script expects a list of examples
            if isinstance(data, list):
                return data
            # Handle cases where the JSON might be a dict containing the list
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            else:
                print(f"Warning: JSON file {path} is not a list of examples.", file=sys.stderr)
                return []
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON file {path}. Malformed JSON.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        return [] # Return an empty list to prevent crashing

def prepare_analysis_prompt(question, schema_text):
    """Constructs the prompt for the model analysis."""
    # This prompt structure should match what the model was trained on
    return f"### SQLite Schema:\n{schema_text}\n\n### Question:\n{question}\n\n### SQL Query:\n"

# ---
# SECTION 2: CORE INTERPRETABILITY LOGIC (POST-HOC ANALYSIS)
# ---

def run_intrinsic_attention_analysis(attentions, input_tokens, question, schema, nlp_model):
    """
    Analyzes the model's learned attention weights. This is a post-hoc process.
    """
    num_layers, num_heads, _, _ = attentions.shape
    
    # Create dependency (M_dep) and schema-linking (M_align) matrices for scoring
    doc = nlp_model(question)
    M_dep = np.zeros((len(input_tokens), len(input_tokens)))
    for token in doc:
        try:
            # The tokenizer often adds a prefix space ' '
            token_idx = input_tokens.index(f" {token.text}")
            head_idx = input_tokens.index(f" {token.head.text}")
            if token.head != token: M_dep[token_idx, head_idx] = 1
        except ValueError: continue # Token not found

    M_align = np.zeros((len(input_tokens), len(input_tokens)))
    q_toks = {f" {t.lower()}" for t in question.split()}
    s_toks = {f" {t.lower()}" for t in schema.split()}
    for i, tok_i in enumerate(input_tokens):
        for j, tok_j in enumerate(input_tokens):
            if (tok_i.lower() in q_toks and tok_j.lower() in s_toks) or \
               (tok_j.lower() in q_toks and tok_i.lower() in s_toks):
                M_align[i, j] = 1

    # Score and classify each head based on its learned behavior
    head_scores = [{'layer': l, 'head': h, 
                    's_syntax': np.sum(attentions[l, h] * M_dep),
                    's_link': np.sum(attentions[l, h] * M_align)}
                   for l in range(num_layers) for h in range(num_heads)]
    
    df = pd.DataFrame(head_scores)
    H_syn = df[df['s_syntax'] >= df['s_syntax'].quantile(0.90)]
    H_link = df[df['s_link'] >= df['s_link'].quantile(0.90)]

    # Aggregate matrices and calculate centrality
    A_total = attentions.mean(axis=(0, 1))
    A_syn = attentions[H_syn['layer'].values, H_syn['head'].values].mean(axis=0) if not H_syn.empty else np.zeros_like(A_total)
    A_link = attentions[H_link['layer'].values, H_link['head'].values].mean(axis=0) if not H_link.empty else np.zeros_like(A_total)

    def get_centrality(matrix):
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        try:
            eigen_cen = nx.eigenvector_centrality_numpy(G, weight='weight')
        except: # Fallback for convergence issues
            eigen_cen = {i: 0 for i in range(matrix.shape[0])}
        degree_cen = {n: v for n, v in G.degree(weight='weight')}
        return eigen_cen, degree_cen

    eigen_total, degree_total = get_centrality(A_total)
    eigen_syn, _ = get_centrality(A_syn)
    eigen_link, _ = get_centrality(A_link)
    
    scores = {
        'Eigenvector (Total)': eigen_total, 'Degree (Total)': degree_total,
        'Eigenvector (Syntax)': eigen_syn, 'Eigenvector (Link)': eigen_link,
    }
    graphs = {'Total': A_total, 'Syntax': A_syn, 'Link': A_link}
    return scores, graphs

def run_post_hoc_gradient_analysis(model, tokenizer, inputs, sql_truth):
    """
    Runs Saliency and Integrated Gradients using Captum for comparison.
    Moves model to CPU to avoid CUDA OOM errors during this intensive step.
    """
    # *** BUG FIX IS HERE: Move model to CPU for this memory-heavy step ***
    print("Moving model to CPU for gradient analysis to conserve VRAM...")
    unwrapped_model = model.module if hasattr(model, 'module') else model
    unwrapped_model.to('cpu')
    
    def model_forward(inputs_embeds):
        outputs = unwrapped_model(inputs_embeds=inputs_embeds)
        target_token_str = sql_truth.split()[0] if sql_truth else ""
        if not target_token_str: 
            return torch.tensor([0.0])
        
        target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]
        target_logit = outputs.logits[0, -1, target_token_id]
        return target_logit.unsqueeze(0)

    # Ensure inputs are on the CPU as well
    input_embeddings = unwrapped_model.get_input_embeddings()(inputs['input_ids'].to('cpu'))
    baseline = torch.zeros_like(input_embeddings)

    # Run attribution on the CPU
    saliency_attr = Saliency(model_forward).attribute(input_embeddings)
    ig_attr = IntegratedGradients(model_forward).attribute(input_embeddings, baselines=baseline)

    saliency_scores = saliency_attr.norm(dim=-1).squeeze(0).cpu().numpy()
    ig_scores = ig_attr.norm(dim=-1).squeeze(0).cpu().numpy()
    
    return {'Saliency': saliency_scores, 'Integrated Gradients': ig_scores}


def save_analysis_artifacts(df, graphs, tokens, question_text, output_dir, ex_idx):
    """Saves the results DataFrame to CSV and graph visualizations to PNG."""
    # Save scores
    csv_path = os.path.join(output_dir, f"example_{ex_idx}_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"✔️ Saved detailed scores to {csv_path}")

    # Save graph visualizations
    q_toks = {f" {t.lower()}" for t in question_text.split()}
    node_colors = ['skyblue' if t.lower() in q_toks else 'lightgreen' for t in tokens]
    
    for name, matrix in graphs.items():
        filepath = os.path.join(output_dir, f"example_{ex_idx}_graph_{name.lower()}.png")
        plt.figure(figsize=(22, 22))
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
        edge_widths = [d['weight'] * 8 for _, _, d in G.edges(data=True)]

        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color='grey')
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors)
        nx.draw_networkx_labels(G, pos, labels={i: t for i, t in enumerate(tokens)}, font_size=11)
        
        plt.title(f"{name} Attention Graph (Example {ex_idx})", size=26)
        plt.savefig(filepath)
        plt.close()
    print(f"✔️ Saved graph visualizations for example {ex_idx}")

# ---
# SECTION 3: MAIN EXECUTION SCRIPT
# ---

def main():
    parser = argparse.ArgumentParser(description="Analyze a trained LLaMA checkpoint for Text-to-SQL interpretability.")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, 
        help="Path to the directory containing the trained model checkpoint (e.g., './llama_finetuned')."
    )
    parser.add_argument("--eval_path", type=str, required=True, help="Path to evaluation data in JSONL format.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Root directory for the SQLite databases.")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Directory to save analysis results.")
    parser.add_argument("--num_examples", type=int, default=1, help="Number of examples from the eval set to analyze.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--- Step 1: Loading Trained Model and Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=False)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" 
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # We will only wrap with DataParallel if we are NOT moving the model to CPU later
    # For this version, we will manage device placement manually.
    is_multi_gpu = torch.cuda.device_count() > 1
    if is_multi_gpu:
        print(f"✔️ {torch.cuda.device_count()} GPUs detected. Will manage device placement manually.")

    nlp = spacy.load("en_core_web_sm")
    print(f"✔️ Successfully loaded model from: {args.checkpoint_path}")

    print("\n--- Step 2: Running Analysis on Evaluation Data ---")
    eval_data = load_evaluation_data(args.eval_path)
    
    for i, ex in enumerate(eval_data[:args.num_examples]):
        ex_idx = i + 1
        print(f"\n{'='*30} Analyzing Example {ex_idx} {'='*30}")
        db_id, question, sql_truth = ex.get("db_id"), ex.get("question"), ex.get("SQL", "")
        
        db_path = os.path.join(args.db_root_path, db_id, f"{db_id}.sqlite")
        schema_text = get_schema_from_db(db_path)
        prompt_text = prepare_analysis_prompt(question, schema_text)
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Get the learned attention weights from the model forward pass (on GPU)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        attentions = torch.stack(outputs.attentions).squeeze(1).cpu().float().numpy()

        # Run intrinsic analysis (on CPU with numpy)
        intrinsic_scores, graphs = run_intrinsic_attention_analysis(attentions, input_tokens, question, schema_text, nlp)
        
        # Memory Management
        del outputs, attentions
        gc.collect()
        torch.cuda.empty_cache()
        
        # Run gradient-based analysis (on CPU)
        gradient_scores = run_post_hoc_gradient_analysis(model, tokenizer, inputs, sql_truth)
        
        # Move model back to GPU for the next example's attention analysis
        model.to(device)

        # Combine all scores into a single DataFrame for reporting
        df = pd.DataFrame(index=range(len(input_tokens)))
        df['Token'] = input_tokens
        for name, scores in {**intrinsic_scores, **gradient_scores}.items():
            df[name] = df.index.map(scores.get) if isinstance(scores, dict) else scores
        
        # Normalize scores (0-1) for easier comparison across methods
        for col in df.columns.drop('Token'):
            if df[col].max() > 0:
                df[col] = df[col] / df[col].max()

        print(f"\n--- Analysis Results for Example {ex_idx} ---\nQuestion: {question}")
        print(df.round(4).to_string())
        
        save_analysis_artifacts(df, graphs, input_tokens, question, args.output_dir, ex_idx)
        
        # Memory Management
        del intrinsic_scores, graphs, gradient_scores, df, inputs, input_tokens
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()






