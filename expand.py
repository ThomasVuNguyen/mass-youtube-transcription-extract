#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Expansion Script

This script takes a dataset in Alpaca JSON format (instruction, input, output)
and expands it by generating new, similar entries using Cloudflare Workers AI.
It aims to increase the dataset size by a specified factor (e.g., 4x).
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Any, Optional
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default Cloudflare Worker AI model
DEFAULT_MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct"

class CloudflareAI:
    """
    A class to interact with Cloudflare Workers AI API.
    """
    
    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.rate_limit_delay = 1 # seconds, adjust as needed

    def generate_text(self, model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate text using Cloudflare Workers AI.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant specializing in rephrasing questions and generating thoughtful, philosophical responses."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/{model}",
                    headers=self.headers,
                    json=payload,
                    timeout=60 # Increased timeout
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                
                result = response.json()
                if result.get("success", False) and result["result"] and "response" in result["result"]:
                    return result["result"]["response"].strip()
                else:
                    print(f"AI API Error: {result.get('errors', 'Unknown error')}")
                    if attempt < retries - 1:
                        time.sleep(self.rate_limit_delay * (2 ** attempt)) # Exponential backoff
                    else:
                        return "" # Return empty if all retries fail
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                if attempt < retries - 1:
                    time.sleep(self.rate_limit_delay * (2 ** attempt))
                else:
                    return ""
            time.sleep(self.rate_limit_delay) # Basic rate limiting
        return ""

def rephrase_input(original_input: str, cf_ai: CloudflareAI, model: str) -> Optional[str]:
    """
    Rephrases the original input question using AI to create a similar variation.
    """
    prompt = (
        f"Please rephrase the following question or statement. The rephrased version should maintain the original core meaning and intent, "
        f"but be worded differently. It should sound natural and be suitable to elicit a thoughtful, philosophical response.\n\n"
        f"Original: \"{original_input}\"\n\n"
        f"Rephrased version (provide only the rephrased text, no extra commentary):"
    )
    rephrased = cf_ai.generate_text(model, prompt, max_tokens=150, temperature=0.8)
    return rephrased if rephrased and rephrased.lower() != original_input.lower() else None

def generate_new_output(instruction: str, new_input: str, cf_ai: CloudflareAI, model: str, original_output_hint: Optional[str] = None) -> Optional[str]:
    """
    Generates a new output (response) based on the instruction and the new (rephrased) input.
    Optionally uses the original output as a hint for style/topic but aims for a new response.
    """
    hint_text = ""
    if original_output_hint:
        hint_text = f"For context, a previous response to a similar query was along the lines of: \"{original_output_hint[:200]}...\". Generate a new, distinct response in a similar philosophical and thoughtful style."

    prompt = (
        f"{instruction}\n\n"
        f"Input/Problem: \"{new_input}\"\n\n"
        f"{hint_text}\n\n"
        f"Please provide a new, thoughtful, and well-structured response. Ensure it is coherent, complete, and avoids simple platitudes. "
        f"The response should offer genuine perspective or wisdom."
    )
    new_output = cf_ai.generate_text(model, prompt, max_tokens=500, temperature=0.75)
    # Basic check to ensure it's not just repeating the input or hint
    if new_output and original_output_hint and new_output.strip().lower() in original_output_hint.lower():
        return None 
    return new_output if new_output else None

def main():
    parser = argparse.ArgumentParser(description="Expand an Alpaca-formatted JSON dataset using Cloudflare AI.")
    parser.add_argument("input_json", type=str, help="Path to the input JSON file (Alpaca format).")
    parser.add_argument("output_json", type=str, help="Path to save the expanded JSON dataset.")
    parser.add_argument("--api-token", type=str, default=os.environ.get("CLOUDFLARE_API_TOKEN"),
                        help="Cloudflare API token (defaults to CLOUDFLARE_API_TOKEN env variable).")
    parser.add_argument("--account-id", type=str, default=os.environ.get("CLOUDFLARE_ACCOUNT_ID"),
                        help="Cloudflare account ID (defaults to CLOUDFLARE_ACCOUNT_ID env variable).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Cloudflare Workers AI model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--expansion_factor", type=int, default=4, 
                        help="Desired total size factor (e.g., 4 for 4x the original size). Minimum 2.")
    parser.add_argument("--max_entries", type=int, default=None,
                        help="Maximum number of original entries to process (for testing/limiting API calls).")

    args = parser.parse_args()

    if args.expansion_factor < 2:
        print("Error: Expansion factor must be at least 2.")
        return

    if not os.path.isfile(args.input_json):
        print(f"Error: Input file not found: {args.input_json}")
        return

    api_token = args.api_token
    account_id = args.account_id

    if not api_token or not account_id:
        print("Error: Cloudflare API token and account ID are required.")
        print("Please provide them via arguments or set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID environment variables.")
        return

    cf_ai = CloudflareAI(api_token, account_id)

    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input_json}. Please ensure it's a valid JSON file.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    if not isinstance(original_data, list):
        print("Error: Input JSON must be a list of objects.")
        return

    expanded_data = []
    entries_to_process = original_data
    if args.max_entries is not None and args.max_entries > 0:
        entries_to_process = original_data[:args.max_entries]
        print(f"Processing a maximum of {args.max_entries} original entries.")

    num_new_entries_per_original = args.expansion_factor - 1

    for original_entry in tqdm(entries_to_process, desc="Expanding dataset"):
        if not all(k in original_entry for k in ["instruction", "input", "output"]):
            print(f"Skipping entry due to missing keys: {original_entry}")
            continue

        # Add the original entry first
        expanded_data.append(original_entry)

        successful_generations = 0
        generation_attempts = 0
        # Aim for expansion_factor - 1 new entries
        while successful_generations < num_new_entries_per_original and generation_attempts < num_new_entries_per_original * 2:
            generation_attempts += 1
            # 1. Rephrase input
            rephrased_input = rephrase_input(original_entry["input"], cf_ai, args.model)
            if not rephrased_input:
                print(f"Failed to rephrase input for: {original_entry['input'][:50]}...")
                time.sleep(1) # Small delay if rephrasing fails
                continue

            # 2. Generate new output based on original instruction and rephrased input
            new_output = generate_new_output(original_entry["instruction"], rephrased_input, cf_ai, args.model, original_entry["output"])
            if not new_output:
                print(f"Failed to generate new output for rephrased input: {rephrased_input[:50]}...")
                time.sleep(1) # Small delay if generation fails
                continue
            
            new_entry = {
                "instruction": original_entry["instruction"],
                "input": rephrased_input,
                "output": new_output
            }
            expanded_data.append(new_entry)
            successful_generations += 1
            print(f"Successfully generated variation {successful_generations}/{num_new_entries_per_original} for input: {original_entry['input'][:30]}...")

        if successful_generations < num_new_entries_per_original:
            print(f"Warning: Only generated {successful_generations} new entries instead of {num_new_entries_per_original} for one original entry.")

    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(expanded_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully expanded dataset to {len(expanded_data)} entries.")
        print(f"Saved to {args.output_json}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
