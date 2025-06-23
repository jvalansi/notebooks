import pandas as pd
import requests
import csv


def load_ideology_mapping(csv_file_path='ideology_spectrum_mapping.csv'):
    """Load ideology to political spectrum mapping from CSV file."""
    ideology_to_spectrum = {}
    
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ideology = row['Ideology'].strip()
            category = row['Political Spectrum Category'].strip()
            ideology_to_spectrum[ideology] = category
    
    return ideology_to_spectrum


def fetch_political_data():
    """Fetch political data from Wikidata using SPARQL query."""
    # Define the SPARQL query
    query = """
    SELECT ?countryLabel ?partyLabel ?ideologyLabel ?startDate
    WHERE {
      ?country wdt:P31 wd:Q6256;
               p:P6 ?statement.
      ?statement ps:P6 ?headOfGovernment; pq:P580 ?startDate.
      ?headOfGovernment wdt:P102 ?party.
      
      OPTIONAL {
        ?party wdt:P1142 ?ideology.
      }
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """
    
    # Send the request to Wikidata
    url = 'https://query.wikidata.org/sparql'
    headers = {'User-Agent': 'YourAppName/1.0 (YourContactInfo)'}
    response = requests.get(url, headers=headers, params={'query': query, 'format': 'json'})
    
    return response


def process_political_data(response, ideology_to_spectrum):
    """Process the raw political data response and return structured data."""
    ideologies = []
    country_to_leaning_by_year = []
    
    if response.status_code == 200:
        data = response.json().get('results', {}).get('bindings', [])
        
        # Extract and organize the data
        for entry in data:
            ideology = entry.get('ideologyLabel', {}).get('value', 'Unknown')
            leaning = ideology_to_spectrum.get(ideology, "Unknown")
            
            if leaning in ["Varies", "Unknown", "Unclassified"]:
                continue
                
            result = {
                "country": entry['countryLabel']['value'],
                "party": entry['partyLabel']['value'],
                "start_date": entry['startDate']['value'],
                "ideology": ideology,
                "political_leaning": leaning,
            }
            
            ideologies.append(ideology)
            country_to_leaning_by_year.append(result)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    return ideologies, country_to_leaning_by_year


def save_political_data(country_to_leaning_by_year, output_file='political_data.csv'):
    """Convert political data to DataFrame and save to CSV."""
    political_data = pd.DataFrame(country_to_leaning_by_year)
    
    # Ensure start_date is datetime
    political_data['start_date'] = pd.to_datetime(political_data['start_date'])
    political_data = political_data.sort_values(by='start_date', ascending=False)
    
    print(political_data.head())
    political_data.to_csv(output_file, sep='\t')
    
    return political_data


def collect_and_save_political_data(ideology_csv_path='ideology_spectrum_mapping.csv', 
                                   output_file='political_data.csv'):
    """Main function to collect and save political data."""
    print("Loading ideology mapping...")
    ideology_to_spectrum = load_ideology_mapping(ideology_csv_path)
    
    print("Fetching political data from Wikidata...")
    response = fetch_political_data()
    
    print("Processing political data...")
    ideologies, country_to_leaning_by_year = process_political_data(response, ideology_to_spectrum)
    
    print("Saving political data...")
    political_data = save_political_data(country_to_leaning_by_year, output_file)
    
    print(f"Political data saved to {output_file}")
    return political_data


if __name__ == "__main__":
    collect_and_save_political_data()