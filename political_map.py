import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import requests
from collections import Counter
import csv
import matplotlib.colors as mcolors
import os
from fuzzywuzzy import fuzz, process


# Path to the CSV file
csv_file_path = 'ideology_spectrum_mapping.csv'

# Initialize an empty dictionary to store the mapping
ideology_to_spectrum = {}

# Read the CSV file and populate the dictionary
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        ideology = row['Ideology'].strip()
        category = row['Political Spectrum Category'].strip()
        ideology_to_spectrum[ideology] = category


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

# Ensure the response is successful
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
        # print(f'{country}: {party}, Ideology: {ideology} Leaning: {leaning}')
        ideologies+=[ideology]
        # if leaning not in ["Unknown", "Unclassified", "Varies"]:
        country_to_leaning_by_year += [result]
else:
    print(f"Error: {response.status_code}")
    print(response.text)  # Print response text for debugging

# Convert the country to leaning dictionary into a DataFrame for merging
political_data = pd.DataFrame(country_to_leaning_by_year)
# Ensure start_date is datetime
political_data['start_date'] = pd.to_datetime(political_data['start_date'])
political_data = political_data.sort_values(by='start_date', ascending=False)
print(political_data.head())
political_data.to_csv("political_data.csv", sep='\t')

shp_file_path = os.path.join('data', 'ne_110m_admin_0_countries', 'ne_110m_admin_0_countries.shp')  # Update with your actual path

# Load world shapefile (provided by geopandas)
world = gpd.read_file(shp_file_path)


def preprocess_country_name(country_name):
    # Normalize common variations
    replacements = {
        "United States of America": "United States",
        "Czechia": "Czech Republic",
        # Add any other common aliases here
    }
    return replacements.get(country_name, country_name)

# Function to merge with thresholded fuzzy matching
def merge_with_fuzzy_matching(world_df, leaning_df, score_threshold=80):
    world_df['normalized_admin'] = world_df['ADMIN'].apply(preprocess_country_name)
    matched_countries = []
    
    for country in world_df['normalized_admin']:
        match = process.extractOne(country, leaning_df['country'], scorer=fuzz.ratio)
        # Check if the match score is above the threshold
        if match and match[1] >= score_threshold:
            # if match[1]!=100:
            # print(f"Match found for {country}: {match}")
            matched_countries.append(match[0])
        else:
            matched_countries.append(None)
    
    world_df['matched_country'] = matched_countries
    
    # Merge using the matched countries where a match was found
    return world_df.merge(leaning_df, how='left', left_on='matched_country', right_on='country')

# Merge the datasets
# world = world.merge(political_data, how='left', left_on='ADMIN', right_on='country')
world = merge_with_fuzzy_matching(world, political_data, score_threshold=80)

world.to_csv("world_data.csv", sep='\t')

# Define a gradient color map with appropriate number of colors
color_map = mcolors.LinearSegmentedColormap.from_list(
    'blue_purple_red',
    ['blue', 'purple', 'red']
)

# Assign numerical values to political leanings for mapping
leaning_score = {
    'Far Left': -2,
    'Left': -1.5,
    'Center-Left': -1,
    'Center': 0,
    'Center-Right': 1,
    'Right': 1.5,
    'Far Right': 2
}
world['leaning_score'] = world['political_leaning'].map(leaning_score)

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax)
world.plot(
    column='leaning_score',
    cmap=color_map,
    legend=True,
    legend_kwds={'label': "Political Leaning Spectrum"},
    ax=ax
)

plt.title('World Map: Political Leaning')
plt.show()
