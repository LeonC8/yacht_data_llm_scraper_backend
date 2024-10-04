import urllib.request
import time
import random
from bs4 import BeautifulSoup
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# OpenAI API key setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class YachtData(BaseModel):
    yacht_name: Optional[str]
    brand: Optional[str]
    model: Optional[str]
    price: Optional[str]
    currency_three_letter_shortcut: Optional[str]
    year: Optional[int]
    length: Optional[str]
    length_measurement_unit: Optional[str]
    beam: Optional[str]
    beam_measurement_unit: Optional[str]
    draft: Optional[str]
    draft_measurement_unit: Optional[str]
    num_berths: Optional[int]
    num_heads: Optional[int]
    hull_material: Optional[str]
    engines: Optional[int]
    engine_type: Optional[str]
    fuel_type: Optional[str]
    water_tank_volume: Optional[str]
    water_tank_measurement_unit: Optional[str]
    fuel_tank_volume: Optional[str]
    fuel_tank_measurement_unit: Optional[str]
    location: Optional[str]
    equipment: Optional[List[str]]

def fetch_yacht_listing(url, max_retries=5):
    """Fetch the HTML content of a yacht listing page using requests with retries."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }

    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return None

def process_html_content(html_content):
    """Process the HTML content according to the specified requirements."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract content from the body tag only
    body = soup.find('body')
    if body:
        soup = BeautifulSoup(str(body), 'html.parser')
    
    # Remove all img tags
    for img in soup.find_all('img'):
        img.decompose()
    
    # Remove all a tags completely
    for a in soup.find_all('a'):
        a.extract()
    
    # Remove all input tags
    for input_tag in soup.find_all('input'):
        input_tag.extract()
    
    # Remove all script tags
    for script in soup.find_all('script'):
        script.extract()
    
    # Remove all attributes from remaining tags
    for tag in soup.find_all(True):
        tag.attrs = {}
    
    # Remove all div tags while keeping their content
    for div in soup.find_all('div'):
        div.unwrap()
    
    # Remove all strong tags while keeping their content
    for strong in soup.find_all('strong'):
        strong.unwrap()
    
    # Remove all li tags while keeping their content
    for li in soup.find_all('li'):
        li.unwrap()
    
    # Remove all span tags while keeping their content
    for span in soup.find_all('span'):
        span.unwrap()
    
    # Convert the processed content to a string
    processed_content = str(soup)
    
    # Collapse all newline characters to one
    processed_content = '\n'.join(line.strip() for line in processed_content.splitlines() if line.strip())
    
    return processed_content

def save_processed_content(content, filename):
    """Save the processed content to a file."""
    with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(content)

def read_processed_content(filename):
    """Read the processed content from a file."""
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extract_yacht_data(processed_content):
    """Extract yacht data using GPT-4."""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",  # don't change this model
        messages=[
            {"role": "system", "content": "Extract the yacht information from the HTML content. Dont write the currency, just write the number for the price. The equipments are probably listed under 'Features'"},
            {"role": "user", "content": f"Extract yacht data from this HTML:\n\n{processed_content}"}
        ],
        response_format=YachtData,
    )

    return completion.choices[0].message.parsed

def main():
    # Example yacht listing URL (replace with an actual yacht listing URL)
    url = "https://www.yachtworld.com/yacht/2015-beneteau-gran-turismo-49-fly-9498603/"
    output_filename = "processed_yacht_listing.html" 

    try:
        # Fetch the HTML content
        html_content = fetch_yacht_listing(url)

        if html_content is None:
            print("Failed to fetch the yacht listing.")
            return

        # Process the HTML content
        processed_content = process_html_content(html_content)

        # Save the processed content to a file
        save_processed_content(processed_content, output_filename)

        # Print the number of characters in the processed content
        char_count = len(processed_content)
        print(f"Number of characters in the processed content: {char_count}")
        print(f"Processed content saved to: {output_filename}")

        # Check if the character count is less than 20,000
        if char_count < 40000:
            # Extract yacht data using GPT-4
            yacht_data = extract_yacht_data(processed_content)

            print("\nExtracted Yacht Data:")
            print(yacht_data.model_dump_json(indent=2))

            # Save the extracted data to a file using UTF-8 encoding
            with open("extracted_yacht_data.json", "w", encoding='utf-8') as f:
                f.write(yacht_data.model_dump_json(indent=2))

            print("Extracted data saved to: extracted_yacht_data.json")
        else:
            print(f"Error: Character count ({char_count}) exceeds 40,000. Data not submitted to OpenAI.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()