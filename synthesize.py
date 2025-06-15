#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Style-Guided Synthetic Data Generator

This script takes a JSON file of combined transcripts (style guides) and uses
Cloudflare Workers AI to generate new, original input/output pairs that emulate
the style, tone, and thematic essence of each guide. The 'instruction' field
in the output will be empty. This data can be used for fine-tuning ML models.
"""

import argparse
import json
import os
import random
import re
import time
from typing import Dict, List, Any, Optional, Union
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default Cloudflare Worker AI model
DEFAULT_MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct"
DEFAULT_MODEL = "@cf/meta/llama-3-8b-instruct" # Alternative

class CloudflareAI:
    """
    A class to interact with Cloudflare Workers AI API.
    """
    
    def __init__(self, api_token: str, account_id: str):
        """
        Initialize the CloudflareAI class.
        
        Args:
            api_token (str): Cloudflare API token
            account_id (str): Cloudflare account ID
        """
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, model: str, prompt: str, max_tokens: int = 1024) -> str:
        """
        Generate text using Cloudflare Workers AI.
        
        Args:
            model (str): Model ID to use
            prompt (str): Prompt text to send to the model
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated text
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an AI assistant that analyzes text for its style, tone, and themes, and then generates new, original conversational pairs (an input question/problem and a thoughtful output response) that emulate that style. You will be asked to return these pairs in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{self.base_url}/{model}",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return ""
        
        result = response.json()
        if result.get("success", False):
            return result["result"]["response"]
        else:
            print(f"Error: {result.get('errors', [])}")
            return ""



def load_style_guides(json_file_path: str) -> List[Dict[str, str]]:
    """
    Load style guide transcripts from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing style guides.
        
    Returns:
        List[Dict[str, str]]: List of style guide entries with 'video_id' and 'combined_text'.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Ensure we have a list of dictionaries with the expected structure
        if not isinstance(data, list):
            print(f"Error: Expected a list of style guides, got {type(data).__name__}")
            return []
            
        # Filter out any entries that don't have the expected structure
        valid_guides = []
        for i, guide in enumerate(data):
            if not isinstance(guide, dict):
                print(f"Warning: Skipping non-dictionary item at index {i}")
                continue
                
            if 'combined_text' not in guide or not guide['combined_text']:
                video_id = guide.get('video_id', 'unknown')
                print(f"Warning: Style guide for video {video_id} has no 'combined_text', skipping")
                continue
                
            valid_guides.append({
                'video_id': guide.get('video_id', f'unknown_{i}'),
                'combined_text': guide['combined_text']
            })
            
        print(f"Successfully loaded {len(valid_guides)} valid style guides.")
        return valid_guides
        
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return []
    except Exception as e:
        print(f"Error reading or processing file {json_file_path}: {e}")
        return []

def generate_io_pair_prompt(style_guide_text: str) -> str:
    """
    Creates a prompt to generate a new input/output pair based on a style guide transcript.
    
    Args:
        style_guide_text (str): The transcript text to use as a style guide.
        
    Returns:
        str: The prompt for the AI.
    """
    # Limit length to avoid overly long prompts
    truncated_style_guide_text = style_guide_text[:1500]
    prompt = (
        f"Analyze the following text for its style, tone, thematic focus, and overall conversational pattern. "
        f"The text is a transcript from a short video offering wisdom or perspective.\n\n"
        f"STYLE GUIDE TEXT:\n\"\"\"\n{truncated_style_guide_text}\n\"\"\"\n\n"
        f"Based on your analysis of the STYLE GUIDE TEXT, your task is to generate a COMPLETELY NEW and ORIGINAL conversational pair. "
        f"This pair should consist of:\n"
        f"1. An 'input': A natural, conversational question, problem, or observation that someone might genuinely express (1-3 sentences).\n"
        f"2. An 'output': A thoughtful, wise, and philosophical response to that 'input' (typically 3-7 sentences). The response should offer perspective, guidance, or a deeper understanding, emulating the style, tone, and thematic depth of the STYLE GUIDE TEXT, but must NOT be a direct copy, summary, or minor rephrasing of any part of the STYLE GUIDE TEXT or the generated 'input'. It must be a fresh, creative response.\n\n"
        f"The 'input' and 'output' should feel like they belong together in a coherent conversation, similar to the STYLE GUIDE TEXT's implied dynamic.\n\n"
        f"IMPORTANT QUALITY GUIDELINES for the generated 'input' and 'output':\n"
        f"- Authenticity: Sound like a real, natural conversation.\n"
        f"- Originality: Content must be new, not copied or closely derived from the style guide.\n"
        f"- Coherence: The 'output' should directly and relevantly address the 'input'.\n"
        f"- Structure: Well-formed sentences with proper grammar, punctuation, and capitalization.\n"
        f"- No Fillers: Avoid conversational fillers (e.g., 'um', 'uh', 'like', 'you know').\n"
        f"- Depth: The 'output' should provide genuine insight or perspective, not clichés or superficial advice.\n"
        f"- Style Emulation: Capture the *essence* of the STYLE GUIDE TEXT's communication style (e.g., calm, reflective, direct, metaphorical) without mimicking its specific content.\n\n"
        f"Return your generated pair as a single JSON object with two keys: 'input' and 'output'. For example:\n"
        f'{{\n          "input": "Generated question/problem statement here...",\n          "output": "Generated thoughtful response here..."\n        }}\n'
        f"Do not include any other text, explanations, or apologies outside of this JSON object."
    )
    return prompt

def parse_generated_io_pair(json_string: str) -> Optional[Dict[str, str]]:
    """
    Parses the AI's JSON string response to extract the input/output pair.
    
    Args:
        json_string (str): The JSON string from the AI.
        
    Returns:
        Optional[Dict[str, str]]: A dictionary with 'input' and 'output' keys, or None if parsing fails.
    """
    try:
        # The AI might sometimes wrap the JSON in backticks or add minor text
        match = re.search(r'\{.*?\}', json_string, re.DOTALL)
        if match:
            json_text = match.group(0)
        else:
            json_text = json_string

        data = json.loads(json_text)
        if isinstance(data, dict) and 'input' in data and 'output' in data:
            # Basic validation
            if isinstance(data['input'], str) and data['input'].strip() and \
               isinstance(data['output'], str) and data['output'].strip():
                return {'input': data['input'].strip(), 'output': data['output'].strip()}
            else:
                print(f"Warning: Parsed JSON has missing or empty 'input' or 'output' values. Data: {data}")
                return None
        else:
            print(f"Warning: Parsed JSON does not contain 'input' and 'output' keys. Data: {data}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from AI response: {e}. Response was: {json_string[:500]}...")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}. Response was: {json_string[:500]}...")
        return None



def clean_transcript(transcript: str) -> str:
    """
    Clean up a transcript to make it more suitable for a response.
    This involves removing filler words, ensuring proper sentence casing,
    and standardizing punctuation.
    
    Args:
        transcript (str): The original transcript
        
    Returns:
        str: Cleaned transcript
    """
    # Remove "Synthetic Transcript #N:" or "Transcript #N:" labels
    cleaned = re.sub(r'^(Synthetic Transcript #\d+:|Transcript #\d+:)\s*', '', transcript, flags=re.IGNORECASE).strip()

    # Remove non-speech items like [Music], [Applause], etc.
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    
    # Define a more comprehensive list of filler words and phrases
    filler_words = [
        r'\b(um|uh|umm|uhh|ah|ahh|er|err|hmm|hmmm)\b',
        r'\b(oh okay|okay|so|well|like|you know|I mean|actually|basically|literally|sort of|kind of)\b',
        r'\b(hey man\.|hey man|man:)\b', # Specific to dataset
        r'\b(speaker \d+:)\b'
    ]
    for filler in filler_words:
        cleaned = re.sub(filler, '', cleaned, flags=re.IGNORECASE)
    
    # Normalize spacing: replace multiple spaces with a single space, remove leading/trailing spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Sentence casing: Capitalize the first letter of the text
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    # Capitalize after sentence-ending punctuation (. ! ?)
    cleaned = re.sub(r'([.!?]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), cleaned)
    
    # Ensure the response ends with punctuation if it doesn't already
    if cleaned and cleaned[-1] not in ['.', '!', '?']:
        # If the last part looks like an incomplete sentence, add ellipsis, otherwise a period.
        if len(cleaned.split()) > 3 and cleaned.endswith(('.', '!', '?', ',', ';', ':')):
             pass
        elif len(cleaned.split()) < 5 : # very short, potentially cut off
            cleaned += "..."
        else:
            cleaned += "."
            
    # Remove extra spaces before punctuation
    cleaned = re.sub(r'\s+([,.!?])', r'\1', cleaned)
    
    # Correct common errors like ".. ." -> "..."
    cleaned = cleaned.replace(".. .", "...")
    cleaned = cleaned.replace("...", "...") # ensure single form of ellipsis
    
    # Final strip to catch any leading/trailing spaces introduced
    cleaned = cleaned.strip()
    
    return cleaned



def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description='Generate synthetic training data based on style guide transcripts.')
    parser.add_argument('style_guide_json_path', type=str, help='Path to the JSON file containing style guide transcripts')
    parser.add_argument('output_json_path', type=str, help='Path to save the generated training data JSON file')
    parser.add_argument('--api-token', type=str, default=os.getenv('CLOUDFLARE_API_TOKEN'),
                        help='Cloudflare API token (default: from CLOUDFLARE_API_TOKEN env var)')
    parser.add_argument('--account-id', type=str, default=os.getenv('CLOUDFLARE_ACCOUNT_ID'),
                        help='Cloudflare account ID (default: from CLOUDFLARE_ACCOUNT_ID env var)')
    parser.add_argument('--model', type=str, default='@cf/mistralai/mistral-7b-instruct-v0.1',
                        help='Model to use for generation (default: @cf/mistralai/mistral-7b-instruct-v0.1)')
    parser.add_argument('--num_pairs_per_guide', type=int, default=1,
                        help='Number of input/output pairs to generate per style guide (default: 1)')
    parser.add_argument('--max_guides', type=int, default=None,
                        help='Maximum number of style guides to process (default: all)')
    
    args = parser.parse_args()
    
    # Validate API credentials
    if not args.api_token or not args.account_id:
        print("Error: Missing Cloudflare API credentials. Please provide --api-token and --account-id or set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID environment variables.")
        return
    
    # Check for API token and account ID - get from .env file if not provided
    api_token = args.api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = args.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    
    if not api_token or not account_id:
        print("Error: Cloudflare API token and account ID are required.")
        print("Please provide them via --api-token and --account-id arguments")
        return
    
    # Load style guide transcripts
    print(f"Loading style guide transcripts from {args.style_guide_json_path}...")
    style_guides = load_style_guides(args.style_guide_json_path)
    
    if not style_guides:
        print("No valid style guide transcripts found. Exiting.")
        return
        
    if args.max_guides and args.max_guides > 0:
        style_guides = style_guides[:args.max_guides]
        print(f"Limiting to {len(style_guides)} style guides as specified.")
    
    print(f"Found {len(style_guides)} style guide transcripts to process.")
    
    # Initialize Cloudflare AI
    cf_ai = CloudflareAI(api_token, account_id)
    
    generated_data_pairs = []
    
    print(f"Generating new input/output pairs...")
    for guide in tqdm(style_guides, desc="Processing style guides"):
        style_guide_text = guide['combined_text']
        video_id = guide.get('video_id', 'unknown') # Optional: trace back to style guide
        successful_generations_for_guide = 0
        for i in range(args.num_pairs_per_guide):
            # Create a prompt for generating an I/O pair
            prompt = generate_io_pair_prompt(style_guide_text)
            
            # Generate the I/O pair JSON string
            ai_response_json_str = cf_ai.generate_text(args.model, prompt, max_tokens=700) # Max tokens for input+output
            
            if ai_response_json_str:
                # Parse the I/O pair from the JSON string
                io_pair = parse_generated_io_pair(ai_response_json_str)
                if io_pair:
                    # Clean the generated input and output
                    cleaned_input = clean_transcript(io_pair['input']) # Re-using clean_transcript for general text cleaning
                    cleaned_output = clean_transcript(io_pair['output'])

                    if cleaned_input and cleaned_output:
                        generated_data_pairs.append({
                            "instruction": "", # Instruction is empty as per new requirement
                            "input": cleaned_input,
                            "output": cleaned_output,
                            "style_guide_video_id": video_id # Optional: trace back to style guide
                        })
                        successful_generations_for_guide += 1
                        # print(f"Successfully generated pair {i+1}/{args.num_pairs_per_guide} for style guide {guide.get('video_id', 'N/A')}")
                    else:
                        print(f"Warning: Generated input or output was empty after cleaning for guide {guide.get('video_id', 'N/A')}. Pair: {io_pair}")
                else:
                    print(f"Failed to parse I/O pair from AI response for guide {guide.get('video_id', 'N/A')}. Response: {ai_response_json_str[:100]}...")
            else:
                print(f"Failed to get AI response for style guide {guide.get('video_id', 'N/A')}.")
            
            time.sleep(0.5) # Small delay between generations for a single guide
        
        if successful_generations_for_guide < args.num_pairs_per_guide:
            print(f"Warning: Only generated {successful_generations_for_guide}/{args.num_pairs_per_guide} pairs for style guide {guide.get('video_id', 'N/A')}")
        time.sleep(1) # Delay between processing different style guides
        
    if not generated_data_pairs:
        print("No data was generated.")
        return
    
    print(f"Total new input/output pairs generated: {len(generated_data_pairs)}")
    
    # Save to JSON
    try:
        with open(args.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(generated_data_pairs, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(generated_data_pairs)} entries to {args.output_json_path}")
    except Exception as e:
        print(f"Error saving data to {args.output_json_path}: {e}")

if __name__ == "__main__":
    main()