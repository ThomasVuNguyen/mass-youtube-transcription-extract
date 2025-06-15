#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Synthetic Data Generator for YouTube Shorts Transcripts

This script takes in a transcript JSON file from YouTube shorts and uses
Cloudflare Workers AI to generate synthetic data that emulates the style,
knowledge, and flow of the original content. The synthetic data can be used
for fine-tuning ML models.
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
DEFAULT_MODEL = "@cf/meta/llama-3-8b-instruct"

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

def extract_text_from_transcript(transcript_data: Dict[str, Any]) -> str:
    """
    Extract the text content from a transcript.
    
    Args:
        transcript_data (dict): Transcript data for a single video
        
    Returns:
        str: Combined text from the transcript
    """
    if not transcript_data or "transcript" not in transcript_data or not transcript_data["transcript"]:
        return ""
    
    transcript = transcript_data["transcript"]
    segments = transcript.get("segments", [])
    
    # Extract and combine all text segments
    text = " ".join(segment.get("text", "") for segment in segments)
    return text

def extract_transcripts_from_file(json_file: str) -> List[str]:
    """
    Extract transcripts from a JSON file.
    
    Args:
        json_file (str): Path to JSON file with transcript data
        
    Returns:
        list: List of transcript texts
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the shorts data from the JSON
        shorts_data = data.get('shorts_data', {})
        
        # Extract text from each transcript
        transcripts = []
        for video_id, short_data in shorts_data.items():
            text = extract_text_from_transcript(short_data)
            if text:
                transcripts.append(text)
        
        return transcripts
    
    except Exception as e:
        print(f"Error extracting transcripts: {e}")
        return []

def generate_synthetic_prompt(transcripts: List[str], sample_count: int = 5) -> str:
    """
    Create a prompt for synthetic data generation using multiple transcript samples.
    
    Args:
        transcripts (list): List of transcript texts
        sample_count (int): Number of sample transcripts to include
        
    Returns:
        str: Generated prompt
    """
    # Select a random sample of transcripts if we have more than the sample count
    samples = random.sample(transcripts, min(sample_count, len(transcripts)))
    
    # Create a prompt with examples and instructions
    prompt = (
        "I'll provide you with transcript examples from YouTube shorts. "
        "Based on these examples, generate 10 new similar short transcripts "
        "that imitate the style, tone, knowledge, and flow of these examples. "
        "Each synthetic transcript should be 30-60 seconds in length, authentic, "
        "and follow the same patterns as the examples.\n\n"
        "The YouTube shorts follow this conversation pattern: A person presents a problem or question, "
        "and then there's a thoughtful response that offers perspective or wisdom. "
        "Please maintain this format in your synthetic transcripts.\n\n"
        "EXAMPLES:\n\n"
    )
    
    # Add the sample transcripts
    for i, sample in enumerate(samples):
        prompt += f"Example {i+1}:\n{sample}\n\n"
    
    prompt += (
        "INSTRUCTIONS:\n"
        "1. Create 10 new synthetic transcripts that sound like they could be from the same creator\n"
        "2. Each transcript should be labeled with 'Synthetic Transcript #N:'\n"
        "3. Follow the format where someone presents a problem/question and then receives a response\n"
        "4. Maintain the same style, terminology, and tone as the examples\n"
        "5. Content should be engaging and follow similar topics and philosophical themes\n"
        "6. Each transcript should be coherent and self-contained\n\n"
        "Please generate 10 synthetic transcripts now:"
    )
    
    return prompt

def parse_synthetic_results(generated_text: str) -> List[str]:
    """
    Parse the generated text into individual synthetic transcripts.
    
    Args:
        generated_text (str): Generated text from the AI
        
    Returns:
        list: List of individual synthetic transcripts
    """
    results = []
    
    # Look for patterns like "Synthetic Transcript #1:" or similar
    lines = generated_text.split('\n')
    current_transcript = ""
    
    for line in lines:
        if line.strip().lower().startswith("synthetic transcript #") or line.strip().lower().startswith("transcript #"):
            if current_transcript:
                results.append(current_transcript.strip())
            current_transcript = line + "\n"
        elif current_transcript:
            current_transcript += line + "\n"
    
    # Add the last transcript
    if current_transcript:
        results.append(current_transcript.strip())
    
    return results

def generate_question_for_transcript(transcript: str, cf_ai: 'CloudflareAI', model: str) -> str:
    """
    Generate a natural question that could have prompted the given transcript response.
    
    Args:
        transcript (str): The transcript text to generate a question for
        cf_ai: CloudflareAI instance for API calls
        model: Model to use for generation
        
    Returns:
        str: Generated question
    """
    prompt = f"""Below is a transcript from a YouTube short video that contains wisdom or advice. 
    Generate a natural, conversational question that someone might ask that would prompt this response.
    The question should be brief (1-2 sentences), personal, and sound like something a friend would casually say, 
    expressing a problem, concern, or confusion that the transcript addresses.
    
    Examples of good questions:
    - "Man, I feel like I'm missing out on life. Everyone else seems to be having more fun."
    - "I keep comparing myself to other people and it's making me miserable."
    - "I'm so busy all the time but I don't feel like I'm getting anywhere."
    - "Why do I feel so disconnected from everyone?"
    - "I'm afraid I'll never be good enough."
    
    TRANSCRIPT: {transcript}
    
    QUESTION (respond with just the question, no additional text):"""
    
    # Generate a question using the AI model
    question = cf_ai.generate_text(model, prompt, max_tokens=100)
    
    # Clean up the response if needed
    question = question.strip("\"\'`.,:\n ")
    
    # If the question is too long, truncate it
    if len(question) > 150:
        question = question[:147] + "..."
        
    return question

def clean_transcript(transcript: str) -> str:
    """
    Clean up a transcript to make it more suitable for a response.
    
    Args:
        transcript (str): The original transcript
        
    Returns:
        str: Cleaned transcript
    """
    # Remove non-speech items like [Music], [Applause], etc.
    cleaned = re.sub(r'\[.*?\]', '', transcript)
    
    # Remove speaker indications and fillers if present
    cleaned = re.sub(r'\b(Hey man\.|Hey man|Man:|Speaker \d+:|Um|Uh)\b', '', cleaned)
    
    # Normalize spacing
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def format_for_fine_tuning(synthetic_transcripts: List[str], 
                           original_transcripts: List[str], 
                           cf_ai: 'CloudflareAI',
                           model: str,
                           format_type: str = "instruction",
                           include_originals: bool = True) -> List[Dict[str, str]]:
    """
    Format the synthetic data for fine-tuning.
    
    Args:
        synthetic_transcripts (list): List of synthetic transcripts
        original_transcripts (list): List of original transcripts
        cf_ai: CloudflareAI instance for API calls
        model: Model to use for generation
        format_type (str): Format type ('instruction', 'completion', or 'chat')
        include_originals (bool): Whether to include original transcripts in training data
        
    Returns:
        list: Formatted data for fine-tuning
    """
    formatted_data = []
    
    # Process synthetic transcripts (if any)
    for transcript in synthetic_transcripts:
        cleaned_transcript = clean_transcript(transcript)
        question = generate_question_for_transcript(cleaned_transcript, cf_ai, model)
        
        if format_type == "instruction":
            formatted_data.append({
                "instruction": "Respond to this question or problem with wisdom and perspective:",
                "input": question,
                "output": cleaned_transcript
            })
        elif format_type == "completion":
            formatted_data.append({
                "prompt": f"Question or problem: {question}\nResponse:",
                "completion": cleaned_transcript
            })
        elif format_type == "chat":
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": cleaned_transcript}
                ]
            })
    
    # Process original transcripts
    if include_originals:
        for transcript in tqdm(original_transcripts, desc="Processing original transcripts"):
            cleaned_transcript = clean_transcript(transcript)
            question = generate_question_for_transcript(cleaned_transcript, cf_ai, model)
            
            if format_type == "instruction":
                formatted_data.append({
                    "instruction": "Respond to this question or problem with wisdom and perspective:",
                    "input": question,
                    "output": cleaned_transcript
                })
            elif format_type == "completion":
                formatted_data.append({
                    "prompt": f"Question or problem: {question}\nResponse:",
                    "completion": cleaned_transcript
                })
            elif format_type == "chat":
                formatted_data.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": cleaned_transcript}
                    ]
                })
    
    return formatted_data

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic data from YouTube transcripts for fine-tuning")
    parser.add_argument("transcript_json", type=str, help="Path to JSON file with transcript data")
    parser.add_argument("--output", type=str, default=None, help="Output file path for synthetic data")
    parser.add_argument("--api-token", type=str, default=os.environ.get("CLOUDFLARE_API_TOKEN"), 
                        help="Cloudflare API token (defaults to CLOUDFLARE_API_TOKEN env variable)")
    parser.add_argument("--account-id", type=str, default=os.environ.get("CLOUDFLARE_ACCOUNT_ID"), 
                        help="Cloudflare account ID (defaults to CLOUDFLARE_ACCOUNT_ID env variable)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Cloudflare Workers AI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--format", type=str, default="instruction", choices=["instruction", "completion", "chat"], 
                        help="Output format for fine-tuning (default: instruction)")
    parser.add_argument("--batches", type=int, default=5, help="Number of batches to generate (default: 5)")
    parser.add_argument("--exclude-originals", action="store_true", help="Exclude original transcripts from training data")
    args = parser.parse_args()
    
    # Check if transcript file exists
    if not os.path.isfile(args.transcript_json):
        print(f"Error: File not found: {args.transcript_json}")
        return
    
    # Check for API token and account ID - get from .env file if not provided
    api_token = args.api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = args.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    
    if not api_token or not account_id:
        print("Error: Cloudflare API token and account ID are required.")
        print("Please provide them via --api-token and --account-id arguments")
        return
    
    # Extract transcripts from the JSON file
    print(f"Extracting transcripts from {args.transcript_json}...")
    transcripts = extract_transcripts_from_file(args.transcript_json)
    
    if not transcripts:
        print("Error: No valid transcripts found in the file.")
        return
    
    print(f"Found {len(transcripts)} transcripts.")
    
    # Initialize Cloudflare AI
    cf_ai = CloudflareAI(api_token, account_id)
    
    # Generate synthetic data in batches
    all_synthetic = []
    
    print(f"Generating {args.batches} batches of synthetic data...")
    for i in tqdm(range(args.batches)):
        # Create a prompt
        prompt = generate_synthetic_prompt(transcripts)
        
        # Generate synthetic data
        generated_text = cf_ai.generate_text(args.model, prompt)
        
        if not generated_text:
            print(f"Warning: No text generated in batch {i+1}")
            continue
        
        # Parse the results
        synthetic_batch = parse_synthetic_results(generated_text)
        all_synthetic.extend(synthetic_batch)
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    print(f"Generated {len(all_synthetic)} synthetic transcripts.")
    
    # Format for fine-tuning
    formatted_data = format_for_fine_tuning(
        all_synthetic, 
        transcripts,
        cf_ai,
        args.model,
        format_type=args.format,
        include_originals=not args.exclude_originals
    )
    
    # Generate output path if not provided
    if not args.output:
        base_name = os.path.basename(args.transcript_json).split('.')[0]
        args.output = f"{base_name}_synthetic_training_data.json"
    
    # Save to JSON file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Successfully saved {len(formatted_data)} training examples to {args.output}")
    print("\nSample synthetic transcript:")
    if all_synthetic:
        print(all_synthetic[0])

if __name__ == "__main__":
    main()