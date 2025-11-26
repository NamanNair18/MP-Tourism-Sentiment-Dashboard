# Touriscope: Data-Driven Tourism Intelligence Platform (Madhya Pradesh) 🗺️

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini API](https://img.shields.io/badge/Google%20GenAI-1373E6?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/gemini-api)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## 🌟 Project Overview

**Touriscope** is an advanced tourism intelligence platform designed to analyze and translate unstructured social media feedback into **actionable data** for tourism and civic bodies in **Madhya Pradesh (MP)**, the 'Heart of India'.

The platform's core capability is correlating tourist sentiment with specific civic issues (like cleanliness and maintenance) to identify **High-Priority Intervention Areas** where local action will yield the maximum positive impact on the tourist experience and local economy.

---

## ✨ Key Features

The application is structured around two main analytical components and one interactive AI assistant:

### 1. 📊 Sentiment & Geospatial Dashboard
* **Weighted Sentiment Analysis:** Calculates the tourist satisfaction index across over a dozen major MP attractions (e.g., Khajuraho, Ujjain, Bhopal).
* **Geospatial Visualization:** Displays sentiment trends on an interactive map, allowing city-level performance tracking.
* **KPI Tracking:** Monitors total discussion volume, overall sentiment share, and top trending places.

### 2. ⚖️ Integrated Civic Impact Analysis
* **Correlation Engine:** Quantifies the relationship between **low sentiment scores** (negative tourist experience) and the **density of inferred civic complaints** (social media posts mentioning "garbage," "smell," "dirty").
* **Actionable Insights:** Uses a **Dual-Axis Chart** to visually pinpoint cities where fixing cleanliness issues is most critical to boosting tourism satisfaction and civic health.

### 3. 🤖 Gemini-Powered Chatbot Assistant
* **Secure API Integration:** Uses the **Google Gemini API** with Streamlit Secrets for secure key management.
* **Function Calling:** Leverages Gemini's function calling to automatically recognize and **structurally log** real-time tourist complaints (e.g., "The area near the temple is dirty") into a structured data format for immediate processing.
* **Natural Language Q&A:** Answers tourist queries about MP history, timings, and culture using a constrained, data-driven knowledge base.

---

## 💻 Technical Stack & Architecture

| Category | Tools & Libraries Used | Location in Project |
| :--- | :--- | :--- |
| **Programming** | Python 3.8+ | All scripts |
| **Data Processing** | Pandas, NLTK (Text Cleaning) | `data_cleaner.py` |
| **LLM & AI** | Google `google-genai` SDK (Gemini 2.5 Flash) | `app.py` (`get_gemini_response`) |
| **Visualization/UI** | Streamlit, Plotly Express/Graph Objects | `app.py` |
| **Modular Scripts** | Python scripts for structured analysis | `scripts/` folder |

---

## 🚀 Setup and Installation

### Prerequisites
* Python 3.8+
* A Google Gemini API Key.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/Touriscope-Data-Driven-Tourism-Intelligence.git](https://github.com/YourUsername/Touriscope-Data-Driven-Tourism-Intelligence.git)
cd Touriscope-Data-Driven-Tourism-Intelligence