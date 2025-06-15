#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combine Transcript Segments

This script reads a JSON file containing YouTube shorts transcripts (like the
output from a YouTube Transcript API fetcher) and combines the text segments
for each short into a single string. The output is a new JSON file containing
these consolidated transcripts.
"""

import argparse
import json
import os
from typing import Dict, List, Any

def extract_and_combine_text(short_data: Dict[str, Any]) -> str:
    """
    Extracts and combines all text segments from a single short's transcript data.

    Args:
        short_data (dict): The data for a single short, expected to contain
                           a 'transcript' key with 'segments'.

    Returns:
        str: A single string of all combined text segments, or an empty string if
             no valid transcript segments are found.
    """
    if not short_data or "transcript" not in short_data or not short_data["transcript"]:
        return ""
    
    transcript_info = short_data["transcript"]
    if not isinstance(transcript_info, dict):
        # Handle cases where transcript might be a list or other unexpected type
        # Or if it's a list of transcripts, and we need to pick one (e.g., the first)
        # For now, assuming it's a dict or processable as such
        # This part might need adjustment based on exact structure if 'transcript' is not always a dict
        if isinstance(transcript_info, list) and len(transcript_info) > 0:
            # If it's a list of transcripts, let's try to process the first one
            # This is an assumption based on some transcript API outputs
            transcript_info = transcript_info[0]
            if not isinstance(transcript_info, dict):
                 print(f"Warning: Transcript for a short is in an unexpected list format and first item is not a dict.")
                 return ""
        else:
            print(f"Warning: Transcript for a short is not in the expected dictionary format or is an empty list.")
            return ""

    segments = transcript_info.get("segments", [])
    if not segments or not isinstance(segments, list):
        return ""
    
    # Extract and combine all text segments
    text_parts = []
    for segment in segments:
        if isinstance(segment, dict) and "text" in segment and isinstance(segment["text"], str):
            text_parts.append(segment["text"])
    
    return " ".join(text_parts).strip()

def main():
    parser = argparse.ArgumentParser(
        description="Combine transcript text segments from a JSON file."
    )
    parser.add_argument(
        "input_json_path", 
        type=str, 
        help="Path to the input JSON file containing shorts transcript data."
    )
    parser.add_argument(
        "output_json_path", 
        type=str, 
        help="Path to save the output JSON file with combined transcripts."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_json_path):
        print(f"Error: Input file not found: {args.input_json_path}")
        return

    try:
        with open(args.input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input_json_path}. Ensure it's valid.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Assuming the structure from your example and common transcript formats
    # Typically, there's a main key holding all the shorts data, e.g., 'shorts_data'
    # If the root of the JSON is the dictionary of video_id to short_data:
    if isinstance(data, dict) and all(isinstance(v, dict) and 'transcript' in v for v in data.values()):
        shorts_data_map = data
    # Or if it's under a specific key like 'shorts_data' (as in synthesize.py context)
    elif isinstance(data, dict) and 'shorts_data' in data and isinstance(data['shorts_data'], dict):
        shorts_data_map = data['shorts_data']
    else:
        print("Error: Unexpected JSON structure. Expected a dictionary of video_id to short_data, or a dict with a 'shorts_data' key.")
        return

    combined_transcripts_list = []
    for video_id, short_details in shorts_data_map.items():
        combined_text = extract_and_combine_text(short_details)
        if combined_text: # Only add if there's actual text
            combined_transcripts_list.append({
                "video_id": video_id,
                "combined_text": combined_text
            })
        else:
            print(f"Warning: No text extracted for video_id: {video_id}")

    try:
        with open(args.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(combined_transcripts_list, f, indent=2, ensure_ascii=False)
        print(f"Successfully combined transcripts from {len(shorts_data_map)} entries.")
        print(f"{len(combined_transcripts_list)} entries with text saved to {args.output_json_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
