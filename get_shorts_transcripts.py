import argparse
import json
import os
import re
import time
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

def extract_video_id(url):
    """
    Extract YouTube video ID from a URL.
    
    Args:
        url (str): YouTube video URL or shorts URL
        
    Returns:
        str: YouTube video ID or None if not found
    """
    # Pattern to match YouTube video ID in shorts URLs
    # Example: https://www.youtube.com/shorts/abcdef12345
    shorts_pattern = r'/shorts/([a-zA-Z0-9_-]+)'
    
    match = re.search(shorts_pattern, url)
    if match:
        return match.group(1)
    return None

def get_transcript(video_id):
    """
    Get transcript for a YouTube video.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        dict: Transcript information or None if not available
    """
    try:
        # Try to get the transcript
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Get the first available transcript (manual or auto-generated)
        transcript = None
        
        # Try to get a transcript in English first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English transcript not available, get any available transcript
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
            except:
                # Get the first transcript in the list (if any)
                for t in transcript_list:
                    transcript = t
                    break
        
        if transcript:
            # Get the transcript text segments
            transcript_data = transcript.fetch()
            
            # Convert the transcript data to a simple, serializable format
            serializable_segments = []
            for segment in transcript_data:
                # FetchedTranscriptSnippet objects need direct attribute access
                try:
                    serializable_segment = {
                        "text": segment['text'] if isinstance(segment, dict) else segment.text,
                        "start": segment['start'] if isinstance(segment, dict) else segment.start,
                        "duration": segment['duration'] if isinstance(segment, dict) else segment.duration
                    }
                    serializable_segments.append(serializable_segment)
                except (KeyError, AttributeError) as e:
                    print(f"Warning: Unable to process transcript segment: {e}")
                    # Still include partial data if possible
                    partial_segment = {}
                    # Try both dict access and attribute access
                    for attr in ['text', 'start', 'duration']:
                        try:
                            if isinstance(segment, dict):
                                if attr in segment:
                                    partial_segment[attr] = segment[attr]
                            else:
                                if hasattr(segment, attr):
                                    partial_segment[attr] = getattr(segment, attr)
                        except Exception:
                            pass
                    
                    if partial_segment:
                        serializable_segments.append(partial_segment)
                
            return {
                "language": transcript.language,
                "is_generated": transcript.is_generated,
                "segments": serializable_segments
            }
        
    except NoTranscriptFound:
        print(f"No transcript found for video {video_id}")
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video {video_id}")
    except Exception as e:
        print(f"Error getting transcript for video {video_id}: {e}")
    
    return None


def save_transcripts(transcripts_data, output_path=None):
    """
    Save transcripts data to a JSON file.
    
    Args:
        transcripts_data (dict): Transcripts data
        output_path (str, optional): Output file path. If None, will use the channel name.
    
    Returns:
        str: Path to the saved file or None if failed
    """
    if not transcripts_data:
        return None
    
    try:
        channel_name = transcripts_data.get('channel_name', 'unknown')
        
        # Generate output path if not provided
        if not output_path:
            output_path = f"{channel_name}_transcripts.json"
        
        # Make a backup of the file if it exists
        if os.path.exists(output_path):
            backup_path = f"{output_path}.bak"
            try:
                os.replace(output_path, backup_path)
            except Exception as e:
                print(f"Warning: Could not create backup: {e}")
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcripts_data, f, indent=4)
        
        print(f"Successfully saved transcripts to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error saving transcripts: {e}")
        return None

def get_and_save_transcripts_incrementally(json_file, output_path=None, batch_size=5, resume=True):
    """
    Get transcripts for shorts URLs in a JSON file and save them incrementally.
    
    Args:
        json_file (str): Path to JSON file containing shorts URLs
        output_path (str, optional): Output file path. If None, will use the channel name.
        batch_size (int): Number of transcripts to process before saving
        resume (bool): Whether to resume from an existing output file
        
    Returns:
        str: Path to the saved file or None if failed
    """
    try:
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the shorts URLs and channel information
        shorts_urls = data.get('shorts_urls', [])
        channel_name = data.get('channel_name', 'unknown')
        channel_url = data.get('channel_url', '')
        
        if not shorts_urls:
            print(f"No shorts URLs found in {json_file}")
            return None
        
        print(f"Found {len(shorts_urls)} shorts URLs in {json_file}")
        
        # Generate output path if not provided
        if not output_path:
            output_path = f"{channel_name}_transcripts.json"
        
        # Check if we should resume from an existing file
        existing_data = None
        processed_ids = set()
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    
                # Extract already processed video IDs
                if existing_data and 'shorts_data' in existing_data:
                    processed_ids = set(existing_data['shorts_data'].keys())
                    print(f"Found {len(processed_ids)} already processed videos in {output_path}")
            except Exception as e:
                print(f"Error reading existing output file: {e}")
                existing_data = None
                
        # Initialize output data structure
        if not existing_data:
            output_data = {
                "channel_name": channel_name,
                "channel_url": channel_url,
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "shorts_count": len(shorts_urls),
                "transcripts_found": 0,
                "shorts_data": {}
            }
        else:
            output_data = existing_data
            # Update timestamp to reflect the continuation
            output_data["extraction_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S") + " (continued)"
        
        # Counter for batch saving
        batch_counter = 0
        transcripts_found = output_data.get("transcripts_found", 0)
        
        # Get transcripts for each short
        for i, url in enumerate(shorts_urls):
            # Extract video ID
            video_id = extract_video_id(url)
            if not video_id:
                print(f"Could not extract video ID from URL: {url}")
                continue
                
            # Skip if already processed
            if video_id in processed_ids:
                print(f"Skipping {i+1}/{len(shorts_urls)}: {url} (already processed)")
                continue
                
            print(f"Processing {i+1}/{len(shorts_urls)}: {url}")
            
            # Get transcript
            transcript = get_transcript(video_id)
            
            # Store result
            output_data["shorts_data"][video_id] = {
                "url": url,
                "transcript": transcript
            }
            
            # Update transcripts found count
            if transcript is not None:
                transcripts_found += 1
                output_data["transcripts_found"] = transcripts_found
            
            # Increment batch counter
            batch_counter += 1
            
            # Save after each batch
            if batch_counter >= batch_size:
                save_transcripts(output_data, output_path)
                batch_counter = 0
                print(f"Progress saved: {transcripts_found} transcripts found so far")
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Final save if there are any unsaved transcripts
        if batch_counter > 0:
            save_transcripts(output_data, output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        return None

def save_transcripts(transcripts_data, output_path):
    """
    Save transcripts data to a JSON file.
    
    Args:
        transcripts_data (dict): Transcripts data
        output_path (str): Output file path
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcripts_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving transcripts: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract transcripts from YouTube shorts listed in a JSON file.")
    parser.add_argument("json_file", type=str, help="Path to JSON file containing YouTube shorts URLs")
    parser.add_argument("--output", type=str, help="Optional output file path (default: [channel_name]_transcripts.json)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of videos to process before saving (default: 5)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing output file")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        return
    
    # Generate output path if not provided
    if not args.output:
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            channel_name = data.get('channel_name', 'unknown')
            output_path = f"{channel_name}_transcripts.json"
        except Exception as e:
            print(f"Error reading channel name from input file: {e}")
            base_name = os.path.basename(args.json_file).split('.')[0]
            output_path = f"{base_name}_transcripts.json"
    else:
        output_path = args.output
    
    print(f"Transcripts will be saved to: {output_path}")
    
    # Get and save transcripts incrementally
    print(f"Getting transcripts for shorts in {args.json_file}")
    output_file = get_and_save_transcripts_incrementally(
        args.json_file, 
        output_path,
        batch_size=args.batch_size,
        resume=not args.no_resume
    )
    
    if output_file:
        print(f"\nTranscripts saved to {output_file}")
        try:
            # Read the final file to print summary
            with open(output_file, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
                found = final_data.get('transcripts_found', 0)
                total = final_data.get('shorts_count', 0)
                if total > 0:
                    print(f"Summary: Found transcripts for {found} out of {total} shorts ({found/total*100:.1f}%)")
        except Exception as e:
            print(f"Error reading final output file: {e}")
    else:
        print("Failed to save transcripts.")

if __name__ == "__main__":
    main()
