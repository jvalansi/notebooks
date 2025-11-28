import os
import re
from bs4 import BeautifulSoup
import json

def extract_text_from_html(html_file_path):
    """Extract clean text from HTML file"""
    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    except Exception as e:
        print(f"Error processing {html_file_path}: {str(e)}")
        return None

def extract_all_airbnb_texts():
    """Extract text from all HTML files in data/airbnb directory"""
    airbnb_dir = "data/airbnb"
    extracted_texts = {}
    
    if not os.path.exists(airbnb_dir):
        print(f"Directory {airbnb_dir} does not exist")
        return
    
    # Get all HTML files
    html_files = [f for f in os.listdir(airbnb_dir) if f.endswith('.html')]
    
    print(f"Found {len(html_files)} HTML files to process")
    
    for html_file in html_files:
        file_path = os.path.join(airbnb_dir, html_file)
        print(f"Processing: {html_file}")
        
        extracted_text = extract_text_from_html(file_path)
        
        if extracted_text:
            # Use filename without extension as key
            key = os.path.splitext(html_file)[0]
            extracted_texts[key] = {
                'filename': html_file,
                'text': extracted_text,
                'text_length': len(extracted_text)
            }
            print(f"  Extracted {len(extracted_text)} characters")
        else:
            print(f"  Failed to extract text")
    
    return extracted_texts

def save_extracted_texts(extracted_texts, output_format='json'):
    """Save extracted texts to file"""
    if output_format == 'json':
        output_file = 'airbnb_extracted_texts.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_texts, f, indent=2, ensure_ascii=False)
        print(f"Saved extracted texts to {output_file}")
    
    elif output_format == 'txt':
        output_file = 'airbnb_extracted_texts.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for key, data in extracted_texts.items():
                f.write(f"=== {data['filename']} ===\n")
                f.write(f"Length: {data['text_length']} characters\n")
                f.write("-" * 50 + "\n")
                f.write(data['text'])
                f.write("\n" + "=" * 80 + "\n\n")
        print(f"Saved extracted texts to {output_file}")

if __name__ == "__main__":
    print("Extracting text from Airbnb HTML files...")
    
    # Extract texts from all HTML files
    extracted_texts = extract_all_airbnb_texts()
    
    if extracted_texts:
        # Save in both formats
        save_extracted_texts(extracted_texts, 'json')
        save_extracted_texts(extracted_texts, 'txt')
        
        print(f"\nSummary:")
        print(f"Total files processed: {len(extracted_texts)}")
        total_chars = sum(data['text_length'] for data in extracted_texts.values())
        print(f"Total characters extracted: {total_chars:,}")
    else:
        print("No texts were extracted")