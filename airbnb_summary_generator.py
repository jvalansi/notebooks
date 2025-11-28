import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import json

def extract_airbnb_data(html_file_path):
    """Extract structured data from Airbnb HTML file"""
    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        
        # Initialize data dictionary
        data = {
            'option_number': '',
            'filename': os.path.basename(html_file_path),
            'property_name': '',
            'location': '',
            'bedrooms': '',
            'bathrooms': '',
            'guests': '',
            'beds': '',
            'rating': '',
            'reviews_count': '',
            'price_2_nights': '',
            'host_name': '',
            'property_type': '',
            'key_amenities': [],
            'special_features': []
        }
        
        # Extract option number from filename
        option_match = re.search(r'option (\d+)', html_file_path)
        if option_match:
            data['option_number'] = int(option_match.group(1))
        
        # Extract property name (first line after "Show all photos")
        name_match = re.search(r'Show all photos\s*([^\n]+)', text)
        if name_match:
            data['property_name'] = name_match.group(1).strip()
        
        # Extract location (pattern: "Entire home in Location")
        location_match = re.search(r'Entire (?:home|cabin) in ([^0-9]+?)(?:\d|$)', text)
        if location_match:
            data['location'] = location_match.group(1).strip().rstrip(',')
        
        # Extract guest/bedroom/bed/bath info
        guest_info_match = re.search(r'(\d+\+?\s*guests)(\d+\s*bedrooms?)(\d+\s*beds?)(\d+\.?\d*\s*baths?)', text)
        if guest_info_match:
            data['guests'] = guest_info_match.group(1).strip()
            data['bedrooms'] = guest_info_match.group(2).strip()
            data['beds'] = guest_info_match.group(3).strip()
            data['bathrooms'] = guest_info_match.group(4).strip()
        
        # Extract rating and reviews
        rating_match = re.search(r'Rated ([\d\.]+) out of 5 stars', text)
        if rating_match:
            data['rating'] = rating_match.group(1)
        
        reviews_match = re.search(r'(\d+) reviews?', text)
        if reviews_match:
            data['reviews_count'] = reviews_match.group(1)
        
        # Extract price for 2 nights
        price_match = re.search(r'\$([0-9,]+).*?for 2 nights', text)
        if price_match:
            data['price_2_nights'] = f"${price_match.group(1)}"
        
        # Extract host name
        host_match = re.search(r'Hosted by ([^\n]+)', text)
        if host_match:
            data['host_name'] = host_match.group(1).strip()
        
        # Determine property type from filename or content
        if 'cabin' in text.lower() or 'cabin' in html_file_path.lower():
            data['property_type'] = 'Cabin'
        elif 'mansion' in text.lower() or 'mansion' in html_file_path.lower():
            data['property_type'] = 'Mansion'
        else:
            data['property_type'] = 'House'
        
        # Extract key amenities
        amenities = []
        if 'pool' in text.lower():
            amenities.append('Pool')
        if 'hot tub' in text.lower() or 'jacuzzi' in text.lower():
            amenities.append('Hot Tub')
        if 'theater' in text.lower() or 'movie' in text.lower():
            amenities.append('Theater')
        if 'game room' in text.lower() or 'arcade' in text.lower():
            amenities.append('Game Room')
        if 'sauna' in text.lower():
            amenities.append('Sauna')
        if 'wifi' in text.lower():
            amenities.append('WiFi')
        if 'kitchen' in text.lower():
            amenities.append('Kitchen')
        if 'parking' in text.lower():
            amenities.append('Parking')
        if 'pets allowed' in text.lower():
            amenities.append('Pet Friendly')
        
        data['key_amenities'] = amenities
        
        # Extract special features
        features = []
        if 'riverfront' in text.lower() or 'river' in text.lower():
            features.append('Riverfront')
        if 'mountain view' in text.lower():
            features.append('Mountain View')
        if 'ski' in text.lower():
            features.append('Near Skiing')
        if 'wine country' in text.lower():
            features.append('Wine Country')
        if 'lake' in text.lower():
            features.append('Near Lake')
        if 'superhost' in text.lower():
            features.append('Superhost')
        if 'guest favorite' in text.lower():
            features.append('Guest Favorite')
        
        data['special_features'] = features
        
        return data
        
    except Exception as e:
        print(f"Error processing {html_file_path}: {str(e)}")
        return None

def generate_airbnb_summary():
    """Generate summary table from all Airbnb HTML files"""
    airbnb_dir = "data/airbnb"
    
    if not os.path.exists(airbnb_dir):
        print(f"Directory {airbnb_dir} does not exist")
        return
    
    # Get all HTML files and sort by option number
    html_files = [f for f in os.listdir(airbnb_dir) if f.endswith('.html')]
    
    # Sort by option number in filename
    def extract_option_number(filename):
        match = re.search(r'option (\d+)', filename)
        return int(match.group(1)) if match else 999
    
    html_files.sort(key=extract_option_number)
    
    print(f"Processing {len(html_files)} Airbnb listings...")
    
    all_data = []
    
    for html_file in html_files:
        file_path = os.path.join(airbnb_dir, html_file)
        print(f"Processing: {html_file}")
        
        data = extract_airbnb_data(file_path)
        if data:
            all_data.append(data)
    
    if not all_data:
        print("No data extracted")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Clean up and format the data
    df['key_amenities'] = df['key_amenities'].apply(lambda x: ', '.join(x) if x else '')
    df['special_features'] = df['special_features'].apply(lambda x: ', '.join(x) if x else '')
    
    # Reorder columns for better presentation
    column_order = [
        'option_number', 'property_name', 'location', 'property_type', 'guests', 'bedrooms',
        'bathrooms', 'beds', 'rating', 'reviews_count', 'price_2_nights',
        'host_name', 'key_amenities', 'special_features'
    ]
    
    df = df[column_order]
    
    # Save to CSV
    output_file = 'airbnb_summary_table.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSummary table saved to: {output_file}")
    
    # Save detailed data to JSON
    json_output = 'airbnb_detailed_data.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"Detailed data saved to: {json_output}")
    
    # Display summary statistics
    print(f"\n=== AIRBNB LISTINGS SUMMARY ===")
    print(f"Total listings processed: {len(df)}")
    
    # Handle ratings with missing values
    ratings = df['rating'].replace('', None).dropna()
    if len(ratings) > 0:
        avg_rating = ratings.astype(float).mean()
        print(f"Average rating: {avg_rating:.2f} (from {len(ratings)} listings with ratings)")
    else:
        print("Average rating: No ratings available")
    
    # Handle prices with missing values
    prices = df['price_2_nights'].replace('', 'N/A')
    price_list = [p for p in prices if p != 'N/A']
    if price_list:
        print(f"Price range (2 nights): {min(price_list)} - {max(price_list)}")
    else:
        print("Price range: No prices available")
    
    print(f"Property types: {df['property_type'].value_counts().to_dict()}")
    print(f"Locations: {df['location'].value_counts().to_dict()}")
    
    # Display the table
    print(f"\n=== SUMMARY TABLE ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    summary_df = generate_airbnb_summary()