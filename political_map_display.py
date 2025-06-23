import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from fuzzywuzzy import fuzz, process


def load_political_data(data_file='political_data.csv'):
    """Load political data from CSV file."""
    political_data = pd.read_csv(data_file, sep='\t', index_col=0)
    political_data['start_date'] = pd.to_datetime(political_data['start_date'])
    return political_data


def filter_data_by_year(political_data, target_year):
    """Filter political data to get the most recent entry for each country up to target year."""
    # Convert target_year to datetime for comparison
    target_date = pd.to_datetime(f"{target_year}-12-31")
    
    # Make target_date timezone-aware if the data has timezone info
    if political_data['start_date'].dt.tz is not None:
        # If data is timezone-aware, make target_date timezone-aware too
        target_date = target_date.tz_localize('UTC')
    
    # Filter data up to target year
    filtered_data = political_data[political_data['start_date'] <= target_date].copy()
    
    if filtered_data.empty:
        print(f"No data available for year {target_year} or earlier")
        return pd.DataFrame()
    
    # Get the most recent entry for each country up to the target year
    latest_data = filtered_data.sort_values('start_date').groupby('country').tail(1)
    
    print(f"Found data for {len(latest_data)} countries up to year {target_year}")
    return latest_data


def load_world_shapefile(shp_file_path=None):
    """Load world shapefile for mapping."""
    if shp_file_path is None:
        shp_file_path = os.path.join('data', 'ne_110m_admin_0_countries', 'ne_110m_admin_0_countries.shp')
    
    world = gpd.read_file(shp_file_path)
    return world


def preprocess_country_name(country_name):
    """Normalize common country name variations."""
    replacements = {
        "United States of America": "United States",
        "Czechia": "Czech Republic",
        # Add any other common aliases here
    }
    return replacements.get(country_name, country_name)


def merge_with_fuzzy_matching(world_df, leaning_df, score_threshold=80):
    """Merge world shapefile with political data using fuzzy string matching."""
    world_df['normalized_admin'] = world_df['name'].apply(preprocess_country_name)
    matched_countries = []
    
    for country in world_df['normalized_admin']:
        match = process.extractOne(country, leaning_df['country'], scorer=fuzz.ratio)
        # Check if the match score is above the threshold
        if match and match[1] >= score_threshold:
            matched_countries.append(match[0])
        else:
            matched_countries.append(None)
    
    world_df['matched_country'] = matched_countries
    
    # Merge using the matched countries where a match was found
    return world_df.merge(leaning_df, how='left', left_on='matched_country', right_on='country')


def create_color_map():
    """Create color map for political leaning visualization."""
    color_map = mcolors.LinearSegmentedColormap.from_list(
        'blue_purple_red',
        ['blue', 'purple', 'red']
    )
    return color_map


def assign_leaning_scores(world_df):
    """Assign numerical values to political leanings for color mapping."""
    leaning_score = {
        'Far Left': -2,
        'Left': -1.5,
        'Center-Left': -1,
        'Center': 0,
        'Center-Right': 1,
        'Right': 1.5,
        'Far Right': 2
    }
    world_df['leaning_score'] = world_df['political_leaning'].map(leaning_score)
    return world_df


def plot_political_map(world_df, color_map, title='World Map: Political Leaning', 
                      figsize=(15, 10), save_path=None):
    """Create and display the political leaning world map."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot country boundaries
    world_df.boundary.plot(ax=ax)
    
    # Plot colored map based on political leaning
    world_df.plot(
        column='leaning_score',
        cmap=color_map,
        legend=True,
        legend_kwds={'label': "Political Leaning Spectrum"},
        ax=ax
    )
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {save_path}")
    
    plt.show()


def create_political_map(political_data_file='political_data.csv',
                        shp_file_path=None,
                        world_data_output='world_data.csv',
                        score_threshold=80,
                        title='World Map: Political Leaning',
                        figsize=(15, 10),
                        save_path=None,
                        target_year=None):
    """Main function to create and display political map.
    
    Args:
        political_data_file: Path to political data CSV file
        shp_file_path: Path to world shapefile
        world_data_output: Output file for merged world data
        score_threshold: Threshold for fuzzy matching countries
        title: Map title
        figsize: Figure size tuple
        save_path: Path to save the map image
        target_year: Year to filter data up to (if None, uses all data)
    """
    print("Loading political data...")
    political_data = load_political_data(political_data_file)
    
    # Filter data by year if specified
    if target_year is not None:
        print(f"Filtering data up to year {target_year}...")
        political_data = filter_data_by_year(political_data, target_year)
        if political_data.empty:
            print("No data available for the specified year range.")
            return None
        
        # Update title to include year
        if title == 'World Map: Political Leaning':
            title = f'World Map: Political Leaning (up to {target_year})'
    
    print("Loading world shapefile...")
    world = load_world_shapefile(shp_file_path)
    
    print("Merging datasets with fuzzy matching...")
    world = merge_with_fuzzy_matching(world, political_data, score_threshold)
    
    print("Saving merged world data...")
    world.to_csv(world_data_output, sep='\t')
    
    print("Creating color map and assigning scores...")
    color_map = create_color_map()
    world = assign_leaning_scores(world)
    
    print("Plotting political map...")
    plot_political_map(world, color_map, title, figsize, save_path)
    
    return world


def create_political_timeline_sequence(political_data_file='political_data.csv',
                                     shp_file_path=None,
                                     start_year=2000,
                                     end_year=2025,
                                     figsize=(15, 10),
                                     save_directory='political_timeline',
                                     pause_duration=1.0):
    """Create a sequence of political maps showing progression year by year.
    
    Args:
        political_data_file: Path to political data CSV file
        shp_file_path: Path to world shapefile
        start_year: Starting year for timeline
        end_year: Ending year for timeline
        figsize: Figure size tuple
        save_directory: Directory to save timeline maps
        pause_duration: Time to pause between maps (in seconds)
    """
    import os
    import time
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    print(f"Creating political timeline sequence from {start_year} to {end_year}...")
    print("Press Ctrl+C to stop the sequence")
    
    try:
        for year in range(start_year, end_year + 1):
            print(f"\n--- Displaying political map for year {year} ---")
            
            save_path = os.path.join(save_directory, f'political_map_{year}.png')
            world_output = os.path.join(save_directory, f'world_data_{year}.csv')
            
            try:
                # Create and display the map for this year
                world_data = create_political_map(
                    political_data_file=political_data_file,
                    shp_file_path=shp_file_path,
                    world_data_output=world_output,
                    title=f'World Political Leaning - {year}',
                    figsize=figsize,
                    save_path=save_path,
                    target_year=year
                )
                
                if world_data is not None:
                    # Count countries by political leaning for this year
                    leaning_counts = world_data['political_leaning'].value_counts()
                    print(f"Political distribution in {year}:")
                    for leaning, count in leaning_counts.items():
                        if pd.notna(leaning):  # Skip NaN values
                            print(f"  {leaning}: {count} countries")
                
                # Pause before next map
                if year < end_year:  # Don't pause after the last map
                    print(f"Pausing for {pause_duration} seconds...")
                    time.sleep(pause_duration)
                    
            except Exception as e:
                print(f"Error creating map for year {year}: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nSequence interrupted by user")
    
    print(f"\nTimeline sequence complete! Maps saved to {save_directory}/")


def create_political_animation(political_data_file='political_data.csv',
                             shp_file_path=None,
                             start_year=2000,
                             end_year=2025,
                             figsize=(15, 10),
                             save_path='political_animation.gif',
                             duration=1000):
    """Create an animated GIF showing political progression over time.
    
    Args:
        political_data_file: Path to political data CSV file
        shp_file_path: Path to world shapefile
        start_year: Starting year for animation
        end_year: Ending year for animation
        figsize: Figure size tuple
        save_path: Path to save the animated GIF
        duration: Duration of each frame in milliseconds
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import tempfile
    import os
    
    print(f"Creating animated political map from {start_year} to {end_year}...")
    
    # Load data once
    political_data = load_political_data(political_data_file)
    world = load_world_shapefile(shp_file_path)
    color_map = create_color_map()
    
    # Prepare the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    def animate(year):
        ax.clear()
        
        # Filter data for this year
        year_data = filter_data_by_year(political_data, start_year + year)
        
        if year_data.empty:
            ax.text(0.5, 0.5, f'No data available for {start_year + year}',
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title(f'World Political Leaning - {start_year + year}')
            return
        
        # Merge with world data
        world_year = merge_with_fuzzy_matching(world, year_data, 80)
        world_year = assign_leaning_scores(world_year)
        
        # Plot the map
        world_year.boundary.plot(ax=ax, color='black', linewidth=0.5)
        world_year.plot(
            column='leaning_score',
            cmap=color_map,
            legend=True,
            legend_kwds={'label': "Political Leaning Spectrum"},
            ax=ax,
            missing_kwds={'color': 'lightgray'}
        )
        
        ax.set_title(f'World Political Leaning - {start_year + year}', fontsize=16)
        ax.set_axis_off()
    
    # Create animation
    years_range = end_year - start_year + 1
    anim = FuncAnimation(fig, animate, frames=years_range, interval=duration, repeat=True)
    
    # Save as GIF
    print(f"Saving animation to {save_path}...")
    writer = PillowWriter(fps=1000/duration)
    anim.save(save_path, writer=writer)
    
    print(f"Animation saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # Example usage - choose one of the following options:
    
    # Option 1: Create a year-by-year sequence from 2000 to 2025
    print("Creating political timeline sequence from 2000 to 2025...")
    print("This will show maps year by year with a 1-second pause between each.")
    print("Press Ctrl+C to stop the sequence at any time.")
    
    create_political_timeline_sequence(
        start_year=2000,
        end_year=2025,
        pause_duration=1.0,  # 1 second pause between maps
        save_directory='political_timeline_2000_2025'
    )
    
    # Option 2: Create an animated GIF (uncomment to use)
    # print("\nCreating animated GIF...")
    # create_political_animation(
    #     start_year=2000,
    #     end_year=2025,
    #     save_path='political_progression_2000_2025.gif',
    #     duration=1000  # 1 second per frame
    # )
    
    # Option 3: Create individual maps (uncomment to use)
    # print("Creating current political map...")
    # create_political_map()
    #
    # print("\nCreating political map for 2010...")
    # create_political_map(
    #     title='World Political Leaning - 2010',
    #     target_year=2010,
    #     save_path='political_map_2010.png'
    # )