import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
import random 
import google.generativeai as genai
from google.genai.errors import APIError

# --- Configuration (External Data Files) ---
# NOTE: These files must exist in a 'data/' directory relative to app.py
TOURISM_ANALYSIS_FILE = 'data/analysis_results.json'
TOURISM_MAP_FILE = 'data/map_data.json' 
CIVIC_METRICS_FILE = 'data/civic_impact_metrics.json' 

# -----------------------------------------------------
# --- CHATBOT GEMINI CONFIGURATION ---
# -----------------------------------------------------

MODEL_ID = 'gemini-2.5-flash-lite'
# MODEL_ID = 'gemini-2.5-flash' 
# =================================================

# -----------------------------------------------------
# --- CHATBOT MOCK DATA SOURCES  ---
# -----------------------------------------------------

MP_PLACES = {
    "Upper Lake (Bhopal)": {
        "description": "Also known as Bhojtal, it's a massive, beautiful lake central to Bhopal.",
        "timings": "Sunrise to Sunset (6 AM - 7 PM)",
        "rating": "4.6/5.0",
        "special": "Famous for serene boating, spectacular sunset views, and as the lifeblood of the city.",
        "location": "Bhopal."
    },
    "Mahakaleshwar Temple (Ujjain)": {
        "description": "One of the twelve Jyotirlingas, a highly revered Hindu temple dedicated to Lord Shiva.",
        "timings": "4 AM - 11 PM",
        "rating": "4.9/5.0",
        "special": "The temple houses the only Bhasma-Aarti (ash ritual) in the world, performed at dawn.",
        "location": "Ujjain."
    },
    "Western Group Temples (Khajuraho)": {
        "description": "A UNESCO World Heritage site known for intricate, erotic sculptures and magnificent architecture.",
        "timings": "Sunrise to Sunset",
        "rating": "4.8/5.0",
        "special": "Represents the finest art of the Chandela dynasty and medieval Indian art.",
        "location": "Khajuraho."
    },
    "Sarafa Bazaar (Indore)": {
        "description": "A bustling jewelry market by day that transforms into a legendary street food hub at night.",
        "timings": "8 PM - 2 AM (Night Food Market)",
        "rating": "4.7/5.0",
        "special": "Famous for its unique transformation from a gold market to a late-night street food paradise.",
        "location": "Indore."
    },
    "Van Vihar National Park (Bhopal)": {
        "description": "A protected national park and zoo located right next to the Upper Lake.",
        "timings": "7 AM - 6 PM (Varies seasonally)",
        "rating": "4.5/5.0",
        "special": "A refuge for rescued animals, excellent for nature walks, cycling, and bird watching.",
        "location": "Bhopal."
    }
}

TRENDING_DATA = {
    "Mahakaleshwar Temple (Ujjain)": {"sentiment_score": 0.85, "trend": "Trending Positively"},
    "Western Group Temples (Khajuraho)": {"sentiment_score": 0.91, "trend": "Trending Positively"},
    "Upper Lake (Bhopal)": {"sentiment_score": 0.55, "trend": "Holding Neutral"},
    "Sarafa Bazaar (Indore)": {"sentiment_score": 0.45, "trend": "Trending Negatively (Crowding/Cleanliness)"}
}

CULTURAL_FAQ = {
    "food": "Madhya Pradesh offers a diverse cuisine! You must try Indori Poha and Sev, the street food of Sarafa Bazaar, the rich Bhopali Poha, and the traditional Malwa Dal Bafla.",
    "markets": "For traditional handicrafts, visit the **Chowk Bazaar** in Old Bhopal. For the best street food experience, visit **Sarafa Bazaar** in Indore after 8 PM.",
    "history": "MP is the 'Heart of India,' home to UNESCO sites like Khajuraho and Sanchi. It boasts rich histories from the Chandela dynasty to the Mughal Empire, and unique tribal cultures.",
    "best time": "The best time to visit Madhya Pradesh is during the **winter months (October to March)** when the weather is cool and pleasant for sightseeing and safaris.",
    "festivals": "Major festivals celebrated across MP include the Khajuraho Dance Festival, Diwali, and the tribal Bhagoria Festival, all celebrated with local flair and zeal."
}

# In-memory storage for simulated feedback (resets on app reload)
if 'feedback_log' not in st.session_state:
    st.session_state['feedback_log'] = []

# -----------------------------------------------------
# --- 3. CHATBOT CORE LOGIC (Using Gemini 2.5 Flash-Lite) ---
# -----------------------------------------------------

# --- UPDATED SYSTEM INSTRUCTIONS FOR MADHYA PRADESH SCOPE ---
SYSTEM_INSTRUCTIONS = f"""
You are Touriscope Assistant, an AI-powered tourism guide focused on **Madhya Pradesh (MP)**, built for the Touriscope: Data-Driven Tourism Intelligence Platform.
Your purpose is to answer all tourist queries accurately, politely, and using the constraints below.

========================================================
🎯 CONSTRAINTS & BEHAVIOR
========================================================
1.  **Scope:** Focus on tourism, history, culture, and attractions across **all of Madhya Pradesh**.
2.  **Tone:** Maintain a friendly, simple, non-technical, and tourist-focused tone.
3.  **Factual Data:** Never hallucinate or invent factual data not provided below. If information is missing, reply: “Sorry, this information is not available in the Touriscope dataset.”

========================================================
📊 DATA SOURCES (You MUST reference this internal data)
========================================================
-   **Known Attractions:** {json.dumps(MP_PLACES)}
-   **Trending Data:** {json.dumps(TRENDING_DATA)}
-   **Cultural FAQs:** {json.dumps(CULTURAL_FAQ)}

RULES FOR RESPONDING:
-   **Place Queries:** For any place query, include: Description, Timings, Rating, Why it is special, and the **City/Location** (e.g., Ujjain, Khajuraho).
-   **Trending Queries:** Use sentiment info (e.g., “This place is trending positively…”).
-   **Directions:** Give general guidance, NOT live navigation.

========================================================
🗣 FEEDBACK/CLEANLINESS REPORTING (CRITICAL RULE)
========================================================
If a user reports a problem (e.g., "dirty," "garbage," "smell"), you MUST do two things:
1.  **Classify & Structure:** Use Python code to extract and structure the report into a JSON object using this exact schema. Do NOT include the JSON in your final spoken reply.
2.  **Reply:** Always thank the user and reply with: “Thank you! Your feedback has been recorded and forwarded.”

**Feedback JSON Schema:**
{{
"issue_type": "cleanliness / waste / general",
"location": "(location or city mentioned by user or 'unspecified')",
"description": "(short summary of the complaint)",
"user_sentiment": "negative / neutral",
"timestamp": "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
}}
"""

import google.generativeai as genai

def get_gemini_response(prompt: str):
    """Handles Gemini API calls with function calling support, fallback handling, and proper error checks."""

    # 1. Configure API key
    try:
        genai.configure(api_key= st.secrets["gemini"]["api_key"])
    except Exception as e:
        return f"Gemini configuration failed: {e}"

    # 2. Define Python function for feedback logging
    def log_feedback(issue_type: str, location: str, description: str, user_sentiment: str):
        """Logs structured feedback into Streamlit session state."""
        feedback_entry = {
            "Issue Type": issue_type,
            "Location": location.title(),
            "Description": description,
            "User Sentiment": user_sentiment,
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.session_state["feedback_log"].append(feedback_entry)
        return "Feedback logged"

    # 3. Declare function in Gemini tool schema
    tools = [{
        "function_declarations": [
            {
                "name": "log_feedback",
                "description": "Store a cleanliness or civic feedback report",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issue_type": {"type": "string"},
                        "location": {"type": "string"},
                        "description": {"type": "string"},
                        "user_sentiment": {"type": "string"}
                    },
                    "required": ["issue_type", "location", "description", "user_sentiment"]
                }
            }
        ]
    }]

    # 4. Load the model (with fallback if flash-lite fails)
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_ID,
            system_instruction=SYSTEM_INSTRUCTIONS,
            tools=tools
        )
    except:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_INSTRUCTIONS,
            tools=tools
        )

    # 5. Primary model call
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        return f"Gemini error: {e}"

    # 6. Parse tool call (function call)
    try:
        parts = response.candidates[0].content.parts
    except:
        return response.text

    for part in parts:
        if hasattr(part, "function_call"):
            fn = part.function_call

            # Call the Python function dynamically
            if fn.name == "log_feedback":
                result = log_feedback(**fn.args)

                # Send the result back to Gemini for the final message
                followup = model.generate_content(
                    contents=prompt,
                    tool_results=[{
                        "call": fn,
                        "result": result
                    }]
                )
                return followup.text

    return response.text

# -----------------------------------------------------
# --- 4. CHATBOT STREAMLIT UI ---
# -----------------------------------------------------

def render_chatbot_assistant():
    st.title("🤖 Touriscope Assistant: Madhya Pradesh Tourism Chatbot")
    st.markdown("Your AI-powered guide for data-driven insights into Madhya Pradesh's attractions, trends, and tourism intelligence.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm your Touriscope Assistant for **Madhya Pradesh** (The Heart of India). I can answer questions on places like Khajuraho and Ujjain, or log a cleanliness report for you. How can I assist you today?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about places, food, or report a cleanliness issue..."):
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response using the Gemini API
        response = get_gemini_response(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state["messages"].append({"role": "assistant", "content": response})
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Total Feedback Reports Recorded: **{len(st.session_state['feedback_log'])}**")


# -----------------------------------------------------
# --- 5. DASHBOARD LAYOUTS AND UTILITIES ---
# -----------------------------------------------------

def set_custom_styles():
    """Applies custom CSS for better visual appeal and branding, fixing the white text issue."""
    st.markdown(
        """
        <style>
        /* General Font and Spacing */
        .stApp {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Customize the Sidebar */
        .st-emotion-cache-1ldf6w2 {
            background-color: #f0f2f6; /* Light gray sidebar */
            border-right: 2px solid #e0e0e0;
        }
        /* Style for Headers/Titles (Using a strong civic blue) */
        h1, h2, h3 {
            color: #1a629b; 
            padding-top: 10px;
        }
        
        /* FIX: Metric Readability (KPIs) */
        [data-testid="stMetricValue"] {
            color: black !important;
        }
        [data-testid="stMetricLabel"] {
            color: black !important;
        }
        /* Style the Metric Cards box itself */
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e6e6e6;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def load_tourism_data():
    """Loads all data for the Touriscope project."""
    data = {}
    if not os.path.exists(TOURISM_ANALYSIS_FILE) or not os.path.exists(TOURISM_MAP_FILE):
        st.error("Tourism data missing. Run analysis_engine.py and add_coordinates.py.")
        return None
    
    with open(TOURISM_ANALYSIS_FILE, 'r') as f:
        data = json.load(f)
        
    map_data_df = pd.read_json(TOURISM_MAP_FILE)
    map_data_df = map_data_df[map_data_df['city'] != 'MISSING_CITY']
    data['map_data_df'] = map_data_df
    return data

@st.cache_data
def load_civic_metrics():
    """Loads the civic complaint density metrics inferred from tourism data."""
    if not os.path.exists(CIVIC_METRICS_FILE):
        st.error(f"Civic metrics missing. Run civic_complaint_extractor.py.")
        return None
    with open(CIVIC_METRICS_FILE, 'r') as f:
        return json.load(f)

def render_kpi_cards(metrics):
    """Renders the Key Performance Indicator (KPI) cards."""
    st.markdown("### 📊 Key Performance Indicators")
    
    total_posts = metrics['total_posts']
    positive_percent = metrics['sentiment_distribution'].get('Positive', 0)
    top_place = next(iter(metrics['top_10_places'].keys()), "N/A") 
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Posts Analyzed", value=f"{total_posts:,}")
    with col2:
        st.metric(label="Positive Sentiment Share", value=f"{positive_percent}%")
    with col3:
        st.metric(label="Most Discussed Place", value=top_place)
    
    st.markdown("---")

def create_geospatial_map(df):
    """Creates a scatter map of MP showing sentiment by city."""
    if df.empty:
        st.warning("Cannot display map: Geographical data is missing.")
        return

    df['marker_size'] = df['total_posts'].apply(lambda x: 10 + x / 100)
    df['hover_text'] = df.apply(
        lambda row: f"City: {row['city']}<br>Posts: {row['total_posts']}<br>Avg. Sentiment: {row['avg_score']:.3f}", 
        axis=1
    )

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="hover_text",
        size="marker_size", 
        color="avg_score", 
        color_continuous_scale=px.colors.sequential.Inferno, 
        zoom=5.5, 
        center={"lat": 23.00, "lon": 78.00}, 
        title="Geospatial Sentiment Analysis of MP Tourism (Map View)",
        height=600
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


def create_sentiment_distribution_chart(metrics):
    """Creates a donut chart for overall sentiment distribution."""
    sentiment_df = pd.DataFrame(
        list(metrics['sentiment_distribution'].items()), 
        columns=['Sentiment', 'Percentage']
    )
    sentiment_df['Sort_Order'] = sentiment_df['Sentiment'].map({'Positive': 1, 'Neutral': 2, 'Negative': 3})
    sentiment_df.sort_values(by='Sort_Order', inplace=True)
    color_map = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}

    fig = px.pie(
        sentiment_df, names='Sentiment', values='Percentage',
        title='Overall Sentiment Distribution Across Madhya Pradesh Tourism',
        hole=0.4, color='Sentiment', color_discrete_map=color_map,
    )
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


def create_top_places_chart(data):
    """Creates a bar chart for top places by weighted sentiment score."""
    sentiment_df = pd.DataFrame(data['place_sentiment_data'])
    sentiment_df.sort_values(by='Sentiment Index', ascending=False, inplace=True)
    
    fig = px.bar(
        sentiment_df.head(15), 
        x='place_name', y='Sentiment Index', color='Total Posts', 
        color_continuous_scale=px.colors.sequential.Sunset,
        hover_data=['Total Posts'],
        labels={'place_name': 'Attraction/Place Name', 'Sentiment Index': 'Weighted Sentiment Index (0 to 1)'},
        title='Top 15 Places Ranked by Weighted Sentiment Index',
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def render_tourism_dashboard(data):
    """Renders the layout and charts for the Touriscope project."""
    metrics = data['key_metrics']
    map_data_df = data['map_data_df']
    
    st.title("🗺️ Touriscope: Madhya Pradesh Tourism Sentiment Dashboard")
    st.markdown("An analysis of social media posts regarding MP tourism attractions.")
    
    render_kpi_cards(metrics)
    create_geospatial_map(map_data_df)

    colA, colB = st.columns([1, 1.5])
    
    with colA:
        create_sentiment_distribution_chart(metrics)
        
        platform_df = pd.DataFrame(
            list(metrics['platform_distribution'].items()), 
            columns=['Platform', 'Percentage']
        )
        fig_platform = px.pie(
            platform_df, names='Platform', values='Percentage', title='Source Platform Distribution',
            hole=0.4, color='Platform', color_discrete_map={'Twitter': '#1DA1F2', 'Instagram': '#C13584', 'Reddit': '#FF4500'}
        )
        fig_platform.update_traces(textinfo='percent')
        st.plotly_chart(fig_platform, use_container_width=True)


    with colB:
        create_top_places_chart(data)
        
        discussion_df = pd.DataFrame(
            list(metrics['top_10_places'].items()), 
            columns=['Place Name', 'Total Posts']
        ).sort_values(by='Total Posts', ascending=True)

        fig_discussion = px.bar(
            discussion_df, x='Total Posts', y='Place Name', orientation='h',
            title='Top 10 Most Discussed Places (By Post Volume)',
            color='Total Posts', color_continuous_scale=px.colors.sequential.Blues,
        )
        st.plotly_chart(fig_discussion, use_container_width=True)


def render_integrated_analysis(tourism_data, civic_metrics):
    """
    Renders the correlation analysis between tourism sentiment and inferred civic complaints,
    using the Dual-Axis Bar Chart for clarity.
    """
    st.title("🔗 Integrated Civic Impact Analysis: Tourism & Civic Issues")
    st.markdown("This analysis correlates **Average Tourist Sentiment** (low scores = negative experience) with the **Density of Inferred Civic Complaints** (posts mentioning 'garbage', 'smell', 'dirty', etc.) to identify high-impact problem areas.")
    st.markdown("---")

    # 1. Prepare Tourism Data (Weighted Average Sentiment by City)
    tourism_df = tourism_data['map_data_df']
    
    # Calculate a weighted score for robustness
    tourism_df['weighted_score'] = tourism_df['avg_score'] * tourism_df['total_posts']

    tourism_city_metrics = tourism_df.groupby('city').agg(
        total_weighted_score=('weighted_score', 'sum'),
        tourism_posts=('total_posts', 'sum')
    ).reset_index()

    # Calculate the final Weighted Average Sentiment
    tourism_city_metrics['avg_sentiment'] = (
        tourism_city_metrics['total_weighted_score'] / tourism_city_metrics['tourism_posts']
    )
    
    # 2. Prepare Civic Complaint Data
    civic_df = pd.DataFrame(civic_metrics['city_complaint_density'])
    civic_df.rename(columns={'total_civic_complaints': 'inferred_complaints'}, inplace=True)
    
    # 3. Merge and Correlate (on 'city')
    correlation_df = pd.merge(
        tourism_city_metrics,
        civic_df,
        on='city',
        how='inner'
    )
    
    if correlation_df.empty:
        st.warning("No overlapping city data found for correlation. Check data integrity.")
        return

    # Calculate Correlation Coefficient (we expect a negative result)
    correlation_coefficient = correlation_df['avg_sentiment'].corr(correlation_df['inferred_complaints'])

    st.markdown("### Correlation Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Pearson Correlation (Sentiment vs. Inferred Complaints)",
            value=f"{correlation_coefficient:.4f}",
            delta_color="off"
        )
    with col2:
        if correlation_coefficient < -0.5:
            st.error("Strong Negative Correlation Found! 📉")
            st.markdown("This suggests that civic issues like waste management are a **major driver** of negative tourist sentiment in MP.")
        elif correlation_coefficient < 0:
            st.info("Weak Negative Correlation Found.")
            st.markdown("There is a slight link observed; civic issues contribute to, but are not the main driver of, low sentiment.")
        else:
            st.success("Weak/No Correlation Found. 👍")
            st.markdown("Tourist sentiment and civic complaints are independent of each other by location.")
            
    st.markdown("---")
    
    # --- Dual-Axis Bar Chart (Visualization) ---
    st.markdown("### 📈 City-Level Comparison: Sentiment vs. Complaints")
    st.markdown(
        """
        This Dual-Axis chart provides a clear and direct visual comparison to locate **High-Priority Intervention Areas** where civic issues negatively impact tourism.
        
        * **Orange Bars (Left Axis):** Represents the **Density of Inferred Civic Complaints**. **TALL BARS** mean high levels of waste or hygiene complaints.
        * **Blue Line/Markers (Right Axis):** Represents the **Average Tourist Sentiment Score**. **LOW MARKERS** mean a poor tourist experience.
        
        The strongest evidence of correlation is found when a **TALL Orange Bar** aligns with a **LOW Blue Marker** in the same city.
        """
    )


    # Sort the data by Sentiment (ascending) to highlight problem areas first
    correlation_df_sorted = correlation_df.sort_values(by='avg_sentiment', ascending=True)

    # Create figure with secondary y-axis
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Inferred Complaints as a primary bar chart (Orange/Red for "Warning" or "Issue")
    fig_dual.add_trace(
        go.Bar(
            x=correlation_df_sorted['city'],
            y=correlation_df_sorted['inferred_complaints'],
            name='Inferred Civic Complaints (Count)',
            marker_color='#FF4500' 
        ),
        secondary_y=False,
    )

    # Add Average Sentiment as a line/scatter on the secondary axis (Blue for "Analysis")
    fig_dual.add_trace(
        go.Scatter(
            x=correlation_df_sorted['city'],
            y=correlation_df_sorted['avg_sentiment'],
            name='Average Tourist Sentiment (0 to 1)',
            mode='lines+markers',
            marker=dict(size=10, color='#1a629b'), 
            line=dict(width=3, color='#1a629b')
        ),
        secondary_y=True,
    )

    # Add figure title and axis labels
    fig_dual.update_layout(
        title_text="Correlation: Low Sentiment vs. High Complaint Density by City",
        hovermode="x unified",
        height=550
    )

    # Set x-axis title
    fig_dual.update_xaxes(title_text="City / Location", tickangle=-45)

    # Set y-axes titles
    fig_dual.update_yaxes(title_text="Total Inferred Complaints", secondary_y=False, range=[0, correlation_df_sorted['inferred_complaints'].max() * 1.1])
    fig_dual.update_yaxes(title_text="Average Sentiment Score (Higher is Better)", secondary_y=True, range=[0, 1])

    st.plotly_chart(fig_dual, use_container_width=True)

    st.markdown("""
    *Actionable Insight:* The negative correlation is visually apparent where **tall orange bars** coincide with **low blue markers**. These cities (e.g., Ujjain, Indore, if your data shows this) should be prioritized for civic improvement projects to maximize the positive impact on the tourist economy.
    """)
    st.markdown("---")
    st.caption(f"Total Inferred Civic Complaints: {civic_metrics.get('total_extracted_complaints', 0)} / Data source: Filtered tourism data.")
    
def main():
    st.set_page_config(
        page_title="Data Science Project",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_custom_styles() 

    # Sidebar Navigation
    project_mode = st.sidebar.radio(
        "Select Project View:",
        ("Touriscope: MP Tourism Sentiment", "Integrated Civic Impact Analysis", "Touriscope Assistant (Chatbot)")
    )
    

    tourism_data = load_tourism_data()
    civic_metrics = load_civic_metrics()

    if project_mode == "Touriscope: MP Tourism Sentiment":
        if tourism_data:
            render_tourism_dashboard(tourism_data)

    elif project_mode == "Integrated Civic Impact Analysis":
        if tourism_data and civic_metrics:
            render_integrated_analysis(tourism_data, civic_metrics)
        elif not tourism_data or not civic_metrics:
             st.error("Cannot load all data sources. Please ensure all preparation scripts have been run successfully.")
    
    elif project_mode == "Touriscope Assistant (Chatbot)":
        render_chatbot_assistant()


if __name__ == "__main__":
    main()