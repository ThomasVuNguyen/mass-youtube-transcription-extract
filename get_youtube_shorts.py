import argparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import re
import json
from urllib.parse import urlparse
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def extract_shorts_urls_from_html(page_html):
    """Extract YouTube shorts URLs directly from HTML using regex pattern matching.
    
    Args:
        page_html (str): The HTML content of the YouTube shorts page
        
    Returns:
        list: List of extracted YouTube shorts URLs
    """
    try:
        # Extract URLs using regex pattern matching
        urls = []
        
        # Pattern to match YouTube shorts URLs or paths
        # This looks for href attributes containing /shorts/ followed by an ID
        pattern = r'href=["\'](?:https://www\.youtube\.com)?(/shorts/[\w-]+)["\'\s]'
        matches = re.findall(pattern, page_html)
        
        print(f"Found {len(matches)} potential shorts URLs in HTML")
        
        for match in matches:
            # Add domain to relative URLs
            full_url = f'https://www.youtube.com{match}'
                
            # Only add unique URLs
            if full_url not in urls:
                urls.append(full_url)
        
        return urls
    except Exception as e:
        print(f"Error extracting URLs from HTML: {e}")
        return []

def extract_channel_name(channel_url):
    """Extract channel name from YouTube URL.
    
    Args:
        channel_url (str): YouTube channel URL
        
    Returns:
        str: Channel name without @ symbol
    """
    # Parse the URL and extract the path
    parsed_url = urlparse(channel_url)
    path_parts = parsed_url.path.strip('/').split('/')  # ['@channelname', 'shorts']
    
    # The first part should be the channel name if it starts with @
    for part in path_parts:
        if part.startswith('@'):
            # Return without the @ symbol
            return part[1:]
    
    # Fallback: use the entire path if no @ found
    return path_parts[0] if path_parts else "unknown_channel"

def get_shorts_urls(channel_url):
    """Given a YouTube channel URL, navigates to the Shorts tab, scrolls to load all content,
    and uses Gemini API to extract all video short URLs.

    Args:
        channel_url (str): The URL of the YouTube channel.
        api_key (str): The Gemini API key.

    Returns:
        list: A list of URLs for the video shorts found on the channel.
    """
    options = Options()
    options.add_argument('--headless')  # Run in headless mode (no browser window)
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--log-level=3')  # Suppress INFO and WARNING messages from Chrome

    driver = None
    shorts_urls = []

    try:
        # Ensure the channel URL points to the shorts tab
        if '/shorts' not in channel_url:
            if channel_url.endswith('/'):
                channel_url += 'shorts'
            else:
                channel_url += '/shorts'

        print(f"Navigating to: {channel_url}")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(channel_url)

        # Give the page some time to initially load
        time.sleep(5)
        
        # Scroll down to load all shorts (YouTube dynamically loads content on scroll)
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        print("Scrolling to load all shorts...")
        scroll_count = 0
        max_scrolls = 10  # Limit number of scrolls to avoid infinite loop
        
        while scroll_count < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(3)  # Wait for new content to load
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            
            if new_height == last_height:
                print("Reached the end of the page.")
                break
            
            last_height = new_height
            scroll_count += 1
            print(f"Current scroll height: {new_height} (Scroll {scroll_count}/{max_scrolls})")

        # Get the full HTML of the page after scrolling
        page_html = driver.page_source
        print(f"Obtained page HTML, length: {len(page_html)} characters")
        
        # Use regex pattern matching to extract the shorts URLs
        print("Extracting shorts URLs from HTML...")
        shorts_urls = extract_shorts_urls_from_html(page_html)
        
        print(f"Found {len(shorts_urls)} shorts.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:
            driver.quit()
    
    return shorts_urls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch all YouTube Shorts URLs from a given channel.")
    parser.add_argument("channel_url", type=str, help="The URL of the YouTube channel (e.g., https://www.youtube.com/@channelname)")
    parser.add_argument("--output-dir", type=str, default=".", help="Optional directory to save the output JSON file (default: current directory)")
    args = parser.parse_args()
    
    if not args.channel_url.startswith("https://www.youtube.com/"):
        print("Error: Please provide a valid YouTube channel URL starting with 'https://www.youtube.com/'")
        exit(1)
    
    # Extract channel name from URL
    channel_name = extract_channel_name(args.channel_url)
    print(f"Channel name: {channel_name}")
    
    # Generate output filename
    output_file = os.path.join(args.output_dir, f"{channel_name}_shorts.json")
    
    # Get shorts URLs
    urls = get_shorts_urls(args.channel_url)
    
    if urls:
        # Save to JSON file
        try:
            data = {
                "channel_name": channel_name,
                "channel_url": args.channel_url,
                "shorts_count": len(urls),
                "shorts_urls": urls,
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            print(f"\nSuccessfully saved {len(urls)} shorts URLs to {output_file}")
        except Exception as e:
            print(f"Error saving to JSON file: {e}")
            
        # Also print the URLs to console
        print("\n--- Shorts URLs ---")
        for url in urls:
            print(url)
        print(f"\nTotal: {len(urls)} shorts found")
    else:
        print("No shorts URLs found or an error occurred.")
        # Create an empty JSON file
        try:
            data = {
                "channel_name": channel_name,
                "channel_url": args.channel_url,
                "shorts_count": 0,
                "shorts_urls": [],
                "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"Created empty JSON file: {output_file}")
        except Exception as e:
            print(f"Error creating empty JSON file: {e}")
