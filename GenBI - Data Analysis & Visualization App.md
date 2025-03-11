# GenBI - Data Analysis & Visualization App

## Overview
GenBI is a Streamlit-based web application that allows users to upload CSV files and interact with their data through natural language queries. It leverages OpenAI's ChatGPT and Google's Gemini models to perform data analysis and visualization. The app automatically routes user queries to either an analysis agent or a visualization agent based on the context of the request.

## Features
- **Upload and View CSV Data**: Users can upload CSV files and preview the dataset.
- **Natural Language Querying**: Users can input text-based queries to analyze data or generate visualizations.
- **AI-Powered Data Analysis**: Uses OpenAI's ChatGPT or Google's Gemini models to interpret and analyze data.
- **AI-Powered Data Visualization**: Generates plots using Matplotlib and Seaborn.
- **Automated Query Routing**: Directs queries to the appropriate AI agent (Analysis or Visualization).
- **Conversational Memory**: Maintains conversation history to improve interactions.

## Installation
### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8+
- pip package manager

### Steps to Install and Run

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run hackathon.py
   ```

## Usage
1. **Upload CSV**: Click on the "Upload CSV File" button to upload a dataset.
2. **Select Language Model**: Choose between ChatGPT and Gemini.
3. **Interact with Data**: Enter queries such as:
   - "Show summary statistics"
   - "Plot a histogram of column 'Sales'"
   - "Find the correlation between 'Age' and 'Salary'"
4. **View Output**: The app will process the query and return either tabular results or a generated plot.

## API Keys
This application requires API keys for OpenAI and Google Gemini services. Replace the placeholders in the code with valid API keys:
PS: Please replace the API Keys



## Future Enhancements
- Integration with more AI models for enhanced analysis.
- Improved UI for better user experience.


