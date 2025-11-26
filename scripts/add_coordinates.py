import pandas as pd
import os

INPUT_FILE = 'turiscope_mp_tourism_clean_data.csv'

# Dictionary of approximate coordinates for key cities/attractions in MP
COORDINATES_MAP = {
    'Ujjain': (23.1793, 75.7836),
    'Gwalior': (26.2183, 78.1828),
    'Indore': (22.7196, 75.8577),
    'Jabalpur': (23.1815, 79.9865),
    'Bhopal': (23.2599, 77.4126),
    'Khajuraho': (24.8333, 79.9167),
    'Sanchi': (23.4862, 77.7397),
    'Bhedaghat': (23.1415, 79.7891), 
    'Pachmarhi': (22.4697, 78.4357),
    'Mandu': (22.3484, 75.3976),
    'Orchha': (25.3530, 78.6465),
    'Pench National Park': (21.6700, 79.2800),
    'Kanha National Park': (22.3323, 80.5912),
    'MISSING_CITY': (23.00, 78.00)
}

def add_coordinates_and_save():
    """Reads the clean data, adds lat/lon based on city, and overwrites the CSV."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Clean data file not found at {INPUT_FILE}. Please run data_cleaner.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from clean CSV.")

    # Create Latitude and Longitude columns based on the 'city' column
    df['latitude'] = df['city'].apply(lambda x: COORDINATES_MAP.get(x, (None, None))[0])
    df['longitude'] = df['city'].apply(lambda x: COORDINATES_MAP.get(x, (None, None))[1])
    
    # Fill any remaining NaNs (for missing cities not in our map) with the MISSING_CITY coordinates
    default_lat, default_lon = COORDINATES_MAP['MISSING_CITY']
    df['latitude'] = df['latitude'].fillna(default_lat)
    df['longitude'] = df['longitude'].fillna(default_lon)

    # Calculate average sentiment for plotting marker size/color
    avg_sentiment = df.groupby(['city', 'latitude', 'longitude']).agg(
        avg_score=('sentiment_score', 'mean'),
        total_posts=('id', 'count')
    ).reset_index()

    # Save the updated data (with lat/lon) back to the clean CSV
    df.to_csv(INPUT_FILE, index=False)
    print(f"Successfully added coordinates and saved the updated data to {INPUT_FILE}.")
    
    # Save the aggregated map data to a JSON for the app to consume easily
    map_data_output = 'data/map_data.json'
    avg_sentiment.to_json(map_data_output, orient='records', indent=4)
    print(f"Aggregated map data saved to {map_data_output}.")


if __name__ == "__main__":
    add_coordinates_and_save()
    