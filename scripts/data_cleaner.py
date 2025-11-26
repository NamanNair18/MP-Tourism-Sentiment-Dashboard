import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import string
import os

# --- Configuration ---
INPUT_FILE = 'turiscope_mp_tourism_sentiment_dataset_unclean.csv' 
OUTPUT_FILE = 'turiscope_mp_tourism_clean_data.csv'
TEXT_COLUMN = 'text'

# --- Custom Word Lists ---
# Add common Hindi/regional stop words found in the dataset
CUSTOM_NOISE_WORDS = set(['ekdum', 'tha', 'thik', 'thak', 'maja', 'gaya', 'bhi', 'nahi', 'kya'])
# Common hashtag concatenations to fix
SPLIT_PHRASES = {
    'mustvisit': 'must visit',
    'highlyrecommend': 'highly recommend',
    'traveltips': 'travel tips',
    'exploremp': 'explore mp',
}


# --- NLTK Setup ---
def setup_nltk():
    """Downloads the necessary 'stopwords' resource if it hasn't been downloaded."""
    print("Setting up NLTK...")
    try:
        nltk.data.find('corpora/stopwords')
        print("'stopwords' already installed.")
    except LookupError:
        print("Downloading 'stopwords'...")
        nltk.download('stopwords', quiet=True)
        print("'stopwords' downloaded successfully.")
    print("NLTK setup complete.")


# --- Data Cleaning and Type Conversion ---
def clean_and_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs comprehensive cleaning on the DataFrame, including text preprocessing,
    data type conversion, and post-cleaning of categorical and text data.
    """
    print(f"\n--- Starting Data Cleaning and Processing ---")
    initial_rows = len(df)
    
    # 1. Initial Missing Values and Duplicates
    df.dropna(subset=[TEXT_COLUMN], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    rows_dropped = initial_rows - len(df)
    print(f"Dropped {rows_dropped} rows (NaNs in text / Duplicates). Remaining rows: {len(df)}")
    
    # --- A. NUMERICAL CLEANING ---
    for col in ['sentiment_score', 'likes', 'comments']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df[['likes', 'comments']] = df[['likes', 'comments']].fillna(0).astype(int)
    df['sentiment_score'] = df['sentiment_score'].fillna(df['sentiment_score'].median())

    # --- B. TEXT PREPROCESSING (Initial Clean) ---
    df['cleaned_text'] = df[TEXT_COLUMN].astype(str).copy()
    stop_words = set(stopwords.words('english')).union(CUSTOM_NOISE_WORDS)

    def apply_initial_cleaning_steps(text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text) 
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', ' ', text)
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # Simple string split tokenization
        tokens = text.split() 
        
        # Filter English and Custom stopwords
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        text = ' '.join(filtered_tokens)

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['cleaned_text'] = df['cleaned_text'].apply(apply_initial_cleaning_steps)
    
    
    # --- C. POST-PROCESSING (Fixing Remaining Issues) ---
    
    # 2. Impute Missing Categorical Data (city, place_name, tags)
    df['city'] = df['city'].fillna('MISSING_CITY')
    df['place_name'] = df['place_name'].fillna('MISSING_PLACE')
    df['tags'] = df['tags'].fillna('MISSING_TAGS')
    print(f"Imputed NaNs in city/place_name/tags with 'MISSING...'")

    # 3. Fixing Concatenated Words
    def fix_concatenation(text):
        for original, replacement in SPLIT_PHRASES.items():
            text = text.replace(original, replacement)
        return text
        
    df['cleaned_text'] = df['cleaned_text'].apply(fix_concatenation)
    
    print("Post-cleaning steps complete.")
    
    return df

# --- Main Execution Block ---
def main():
    setup_nltk()
    
    # Check if the input file exists (using the unclean file name)
    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Input file '{INPUT_FILE}' not found.")
        print("Please ensure the file is in the correct directory.")
        return

    print(f"\nLoading data from {INPUT_FILE}...")
    try:
        raw_df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    cleaned_df = clean_and_process_data(raw_df.copy()) 
    
    print("\n--- Final Data Check (Types and Sample) ---")
    print("New NaN Counts:")
    print(cleaned_df[['city', 'place_name', 'tags']].isna().sum().to_markdown(numalign="left", stralign="left"))
    
    print("\nSample of Cleaned Data (Showing fixed text and imputed city):")
    # Sort by city to show an imputed row easily, then reset
    print(cleaned_df[['city', 'place_name', 'cleaned_text', 'sentiment_score']].sort_values(by='city', ascending=False).head().to_markdown(index=False, numalign="left", stralign="left"))
    
    print(f"\nExporting cleaned data to {OUTPUT_FILE}...")
    cleaned_df.to_csv(OUTPUT_FILE, index=False)
    
    print("Export complete!")

if __name__ == "__main__":
    main()