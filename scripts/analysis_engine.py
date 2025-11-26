import pandas as pd
import json
import os

# --- Configuration ---
INPUT_FILE = 'turiscope_mp_tourism_clean_data.csv'
OUTPUT_FILE = 'data/analysis_results.json'

# --- Analysis Functions ---

def calculate_place_sentiment(df):
    """
    Calculates the weighted average sentiment score for each unique place.
    Weighting by the number of likes gives more importance to popular posts.
    """
    # NOTE: This function relies on 'sentiment_score' and 'likes' columns.
    # If these are missing from your cleaned CSV, this section will either
    # run with errors or use imputed data (from data_cleaner.py).
    print("Calculating place-specific sentiment scores...")
    
    # 1. Calculate weighted sentiment score (Score * Likes)
    if 'sentiment_score' in df.columns and 'likes' in df.columns:
        df['weighted_score'] = df['sentiment_score'] * df['likes']
        
        # 2. Group by place and sum the weighted score and total likes
        sentiment_metrics = df.groupby('place_name').agg(
            total_weighted_score=('weighted_score', 'sum'),
            total_likes=('likes', 'sum'),
            total_posts=('id', 'count')
        ).reset_index()
        
        # 3. Calculate the Weighted Average Sentiment Score
        sentiment_metrics['weighted_avg_score'] = (
            sentiment_metrics['total_weighted_score'] / sentiment_metrics['total_likes']
        ).fillna(sentiment_metrics['total_weighted_score'] / sentiment_metrics['total_posts']) 
        
        # 4. Filter out the missing place (if the cleaning used imputation)
        sentiment_metrics = sentiment_metrics[
            sentiment_metrics['place_name'] != 'MISSING_PLACE'
        ].sort_values(by='weighted_avg_score', ascending=False)
        
        # Select final columns and rename
        sentiment_metrics = sentiment_metrics[[
            'place_name', 
            'weighted_avg_score', 
            'total_posts'
        ]].rename(columns={'weighted_avg_score': 'Sentiment Index', 'total_posts': 'Total Posts'})
        
        return sentiment_metrics
    else:
        print("Warning: Skipping weighted sentiment calculation due to missing 'sentiment_score' or 'likes'.")
        # Return an empty DataFrame structure for compatibility
        return pd.DataFrame(columns=['place_name', 'Sentiment Index', 'Total Posts'])


def generate_key_metrics(df):
    """
    Calculates overall sentiment distribution and top discussion places.
    """
    metrics = {}

    # 1. Overall Sentiment Distribution
    sentiment_counts = df['sentiment'].value_counts(normalize=True).mul(100).round(2)
    
    # *** CRITICAL FIX HERE: Capitalize keys for dashboard compatibility ***
    metrics['sentiment_distribution'] = {
        k.title(): v 
        for k, v in sentiment_counts.to_dict().items()
    }
    # *******************************************************************

    # 2. Top 10 Most Discussed Places (by total posts)
    top_discussion = df['place_name'].value_counts()
    top_discussion = top_discussion[top_discussion.index != 'MISSING_PLACE']
    metrics['top_10_places'] = top_discussion.head(10).to_dict()
    
    # 3. Platform Distribution
    platform_counts = df['platform'].value_counts(normalize=True).mul(100).round(2)
    metrics['platform_distribution'] = platform_counts.to_dict()

    # 4. Total Posts
    metrics['total_posts'] = int(df.shape[0])
    
    return metrics

# --- Main Execution Block (remains unchanged) ---

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Input file '{INPUT_FILE}' not found.")
        print("Please run data_cleaner.py first, or ensure the file is named correctly.")
        return

    print(f"\nLoading cleaned data from {INPUT_FILE} for analysis...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # --- Run Analysis ---
    
    # Calculate weighted sentiment scores for each place
    place_sentiment_df = calculate_place_sentiment(df)
    
    # Calculate overall key metrics
    key_metrics = generate_key_metrics(df)

    # --- Format Output ---
    
    # Combine all results into a single dictionary
    final_output = {
        'key_metrics': key_metrics,
        'place_sentiment_data': place_sentiment_df.to_dict('records') # List of dictionaries for Streamlit
    }
    
    # --- Export Results ---
    
    print(f"\nExporting analysis results to {OUTPUT_FILE}...")
    
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print("[SUCCESS] Analysis complete! Results saved to analysis_results.json.")
    
    print("\n--- Sample Place Sentiment Data (Top 5) ---")
    print(place_sentiment_df.head().to_markdown(index=False, numalign="left", stralign="left"))


if __name__ == "__main__":
    main()