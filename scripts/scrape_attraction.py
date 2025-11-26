import requests
import pandas as pd
from bs4 import BeautifulSoup 
import os 

def fetch_osm_attractions():
    """Fetches attractions from OpenStreetMap using the Overpass API, covering Madhya Pradesh."""
    overpass_url = "https://overpass-api.de/api/interpreter"

    # Query all tourism-related places in and around Madhya Pradesh
    # The area is set to "Madhya Pradesh"
    query = """
    [out:json][timeout:50];
    area[name="Madhya Pradesh"]->.searchArea;
    (
      node["tourism"](area.searchArea);
      way["tourism"](area.searchArea);
      node["historic"](area.searchArea);
      node["leisure"](area.searchArea);
    );
    out center;
    """

    print("Fetching attractions from OpenStreetMap…")
    try:
        response = requests.get(overpass_url, params={"data": query})
        response.raise_for_status() # Raise an HTTPError for bad responses
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Overpass API: {e}")
        return pd.DataFrame()

    elements = data.get("elements", [])
    print(f"Found {len(elements)} raw places from OpenStreetMap.")

    attractions = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        tourism_type = tags.get("tourism") or tags.get("historic") or tags.get("leisure")
        if name:
            attractions.append({
                "name": name,
                "type": tourism_type,
                "source": "OpenStreetMap",
                "latitude": el.get("lat") or el.get("center", {}).get("lat"),
                "longitude": el.get("lon") or el.get("center", {}).get("lon"),
            })

    return pd.DataFrame(attractions)


def scrape_wikivoyage():
    """
    Placeholder function to scrape attractions from Wikivoyage (https://en.wikivoyage.org/wiki/Bhopal).
    NOTE: This is a placeholder and should be replaced with actual scraping logic 
    targeting relevant Madhya Pradesh pages if expanding the scope beyond Bhopal.
    """
    print("Attempting to scrape attractions from Wikivoyage… (Placeholder)")
    
    # Placeholder data: Replace with real scraping logic
    return pd.DataFrame([
        {"name": "Taj-ul-Masajid", "type": "mosque", "source": "Wikivoyage", "latitude": None, "longitude": None},
        {"name": "Bharat Bhavan", "type": "art_complex", "source": "Wikivoyage", "latitude": None, "longitude": None}
    ])


def scrape_holidify():
    """
    Placeholder function to scrape attractions from Holidify (https://www.holidify.com/places/bhopal/sightseeing-and-things-to-do.html).
    NOTE: This is a placeholder and should be replaced with actual scraping logic 
    targeting relevant Madhya Pradesh pages if expanding the scope beyond Bhopal.
    """
    print("Attempting to scrape attractions from Holidify… (Placeholder)")
    
    # Placeholder data: Replace with real scraping logic
    return pd.DataFrame([
        {"name": "Upper Lake (Bhojtal)", "type": "lake", "source": "Holidify", "latitude": None, "longitude": None},
        {"name": "Van Vihar National Park", "type": "national_park", "source": "Holidify", "latitude": None, "longitude": None}
    ])


def fetch_all_attractions():
    """Combines attractions data from OpenStreetMap, Wikivoyage, and Holidify."""
    
    # Collect DataFrames from all sources
    data_frames = [
        fetch_osm_attractions(),
        scrape_wikivoyage(),
        scrape_holidify()
    ]

    # Concatenate all DataFrames, dropping any empty ones
    all_attractions_df = pd.concat([df for df in data_frames if not df.empty], ignore_index=True)
    
    if all_attractions_df.empty:
        print("\nNo attractions data was successfully fetched from any source.")
        return

    # Clean and deduplicate by name
    df = all_attractions_df.drop_duplicates(subset=["name"])
    
    # Ensure the directory exists before saving (assuming 'data' directory structure)
    os.makedirs(os.path.dirname("data/attractions_raw.csv"), exist_ok=True)
    
    df.to_csv("data/attractions_raw.csv", index=False, encoding="utf-8")
    print(f"\nSuccessfully saved {len(df)} combined and unique Bhopal attractions to data/attractions_raw.csv")

if __name__ == "__main__":
    fetch_all_attractions()