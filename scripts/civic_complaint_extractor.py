import pandas as pd
import json
import os

# --- Configuration ---
INPUT_FILE = 'turiscope_mp_tourism_clean_data.csv' 
OUTPUT_JSON = 'data/civic_impact_metrics.json'
OUTPUT_CSV = 'data/extracted_civic_complaints.csv'

# --- Civic/Waste Keywords ---
# These keywords will be used to identify posts related to waste, dirt, smell, and poor maintenance.
CIVIC_KEYWORDS = [
    'garbage', 'waste', 'smell', 'dirty', 'dustbin', 'unclean', 
    'hygiene', 'litter', 'maintenance', 'filth', 'toilet', 'trash'
]

def extract_and_analyze_civic_data():
    """
    Loads clean tourism data, filters for civic/waste complaints, 
    and calculates complaint density by city.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Clean data file not found at {INPUT_FILE}. Please run data_cleaner.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} cleaned tourism records.")

    # 1. Filter: Posts must have low/negative sentiment
    # We use a threshold of 0.35 to capture Negative and strongly Neutral/Negative posts
    df_filtered = df[df['sentiment_score'] < 0.35].copy()
    print(f"Filtered to {len(df_filtered)} posts with low sentiment.")

    # 2. Filter: Posts must contain at least one civic keyword in the cleaned text
    def contains_civic_keyword(text):
        if pd.isna(text):
            return False
        # Check for presence of any keyword in the cleaned text
        return any(keyword in str(text) for keyword in CIVIC_KEYWORDS)

    df_filtered['is_civic_complaint'] = df_filtered['cleaned_text'].apply(contains_civic_keyword)
    
    # Final dataset of extracted civic complaints
    civic_complaints_df = df_filtered[df_filtered['is_civic_complaint']].copy()
    print(f"Final extracted civic complaints: {len(civic_complaints_df)}.")
    
    if civic_complaints_df.empty:
        print("No civic complaints found with the current filters.")
        return

    # --- Analysis for Correlation ---
    
    # Calculate Civic Complaint Density by City
    city_complaint_density = civic_complaints_df.groupby('city').agg(
        total_civic_complaints=('id', 'count')
    ).reset_index()

    # Save the filtered complaints (optional but good for tracking)
    civic_complaints_df.to_csv(OUTPUT_CSV, index=False)
    
    # Save the density metrics to JSON for the integrated dashboard
    final_output = {
        'total_extracted_complaints': int(len(civic_complaints_df)),
        'city_complaint_density': city_complaint_density.to_dict('records')
    }
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"[SUCCESS] Civic complaint analysis complete. Density metrics saved to {OUTPUT_JSON}.")

if __name__ == "__main__":
    extract_and_analyze_civic_data()