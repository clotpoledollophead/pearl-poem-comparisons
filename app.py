import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Poem Linguistic Analysis", page_icon="📜")

st.title("Comparisons of Poems from the Pearl Manuscript")

# --- Data Loading ---
try:
    df = pd.read_csv("all_poems_analysis_master.csv")
    st.success("Successfully loaded 'all_poems_analysis_master.csv'!")
except FileNotFoundError:
    st.error("Error: 'all_poems_analysis_master.csv' not found. Please ensure that 'articutExtract.py' has been run successfully to generate this file, and it is in the same directory as this Streamlit app.")
    st.stop()

# Ensure relevant columns are strings and strip whitespace
df['Word Type'] = df['Word Type'].astype(str).str.strip()
df['POS_Tag'] = df['POS_Tag'].astype(str).str.strip()

# Get unique poem names
poem_names = df['Poem Name'].unique()

if len(poem_names) < 2:
    st.error(f"Only {len(poem_names)} poem(s) found in the data. Comparison analyses require at least two poems. Please ensure your 'all_poems_analysis_master.csv' contains data for multiple poems.")
    st.stop()

# --- Feature Frequency Calculation (Helper Function) ---
@st.cache_data
def get_feature_frequencies(data_frame, feature_col_display_name):
    """
    Calculates normalized frequency distributions for a given feature column.
    Maps display names ('Word Type', 'POS Tag') to actual column names ('Word Type', 'POS_Tag').
    """
    actual_column_name = feature_col_display_name
    if feature_col_display_name == "POS Tag":
        actual_column_name = "POS_Tag"

    feature_counts = data_frame.groupby('Poem Name')[actual_column_name].value_counts().unstack(fill_value=0)
    feature_counts = feature_counts.fillna(0)
    feature_freq = feature_counts.apply(lambda x: x / x.sum(), axis=1)
    feature_freq = feature_freq.fillna(0)
    return feature_freq

# --- Sidebar for Analysis Options ---
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.radio(
    "Choose Linguistic Feature:",
    ("Word Type", "POS Tag")
)

# Get feature frequencies based on selected analysis type
feature_freq_df = get_feature_frequencies(df, analysis_type)

# --- Section 1: Linguistic Distributions by Poem ---
st.header(f"Linguistic Distributions by Poem ({analysis_type})")
st.markdown("*(Compare the proportion of linguistic features across selected poems)*")

selected_poems_for_dist_chart = st.multiselect(
    "Select Poems to Compare on Bar Chart:",
    poem_names,
    default=list(poem_names[:min(len(poem_names), 2)]), # Default to first two poems
    key='dist_select_poems_multiselect'
)

if len(selected_poems_for_dist_chart) > 0:
    # Filter feature frequency DataFrame for selected poems
    dist_chart_data = feature_freq_df.loc[selected_poems_for_dist_chart].reset_index()

    # Melt the DataFrame to long format for Plotly Express
    dist_chart_data_melted = dist_chart_data.melt(
        id_vars='Poem Name',
        var_name='Feature',
        value_name='Proportion'
    )

    # Filter out features that are 0 for all selected poems to make the chart cleaner
    non_zero_features_dist = dist_chart_data_melted.groupby('Feature')['Proportion'].sum()
    features_to_keep_dist = non_zero_features_dist[non_zero_features_dist > 0].index
    dist_chart_data_melted = dist_chart_data_melted[dist_chart_data_melted['Feature'].isin(features_to_keep_dist)]

    # Sort features for consistent x-axis order (e.g., by overall frequency)
    feature_order_dist = feature_freq_df.sum().sort_values(ascending=False).index.tolist()
    dist_chart_data_melted['Feature'] = pd.Categorical(dist_chart_data_melted['Feature'], categories=feature_order_dist, ordered=True)
    dist_chart_data_melted = dist_chart_data_melted.sort_values('Feature')


    fig = px.bar(
        dist_chart_data_melted,
        x='Feature',
        y='Proportion',
        color='Poem Name', # Differentiate bars by poem
        barmode='group', # Group bars for comparison
        title=f'{analysis_type} Distribution Comparison',
        labels={'Proportion': 'Proportion of Words', 'Feature': analysis_type},
        height=500
    )
    # Improve layout for better readability if many features
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select at least one poem to display the Linguistic Distributions bar chart.")


# --- Section 2: Radar Chart Comparison ---
st.header(f"Radar Chart Comparison of Poems ({analysis_type})")
st.markdown("*(Compare the linguistic profiles of selected poems on key features)*")

selected_poems_for_radar = st.multiselect(
    "Select 2-4 Poems to Compare on Radar Chart:",
    poem_names,
    default=list(poem_names[:min(len(poem_names), 2)]),
    key='radar_select_poems'
)

if len(selected_poems_for_radar) > 0:
    radar_data = feature_freq_df.loc[selected_poems_for_radar].reset_index()
    radar_data_melted = radar_data.melt(id_vars='Poem Name', var_name='Feature', value_name='Proportion')

    non_zero_features = radar_data_melted.groupby('Feature')['Proportion'].sum()
    features_to_keep = non_zero_features[non_zero_features > 0].index
    radar_data_melted = radar_data_melted[radar_data_melted['Feature'].isin(features_to_keep)]

    feature_order = feature_freq_df.sum().sort_values(ascending=False).index.tolist()
    radar_data_melted['Feature'] = pd.Categorical(radar_data_melted['Feature'], categories=feature_order, ordered=True)
    radar_data_melted = radar_data_melted.sort_values('Feature')

    fig_radar = px.line_polar(radar_data_melted,
                              r="Proportion",
                              theta="Feature",
                              color="Poem Name",
                              line_close=True,
                              title=f'Linguistic Profile Comparison ({analysis_type})',
                              height=600,
                              range_r=[0, radar_data_melted['Proportion'].max() * 1.1])
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Please select at least one poem to display the Radar Chart.")

st.markdown("""
### How Visualizations Work:
* **Bar Charts**: Show the proportion of each Word Type or POS Tag for **multiple selected poems**, allowing for direct comparison of their linguistic composition.
* **Radar Chart**: Compares the linguistic profiles of selected poems. Each axis represents a linguistic feature (Word Type or POS Tag), and the lines show the proportion of that feature for each poem, forming a unique "shape" for each poem's profile. This allows for a visual comparison of how different poems utilize various linguistic elements.
""")

st.markdown("---")
st.header("References")
st.markdown("""
Wang, W., Chen, C., Lee, C., Lai, C., & Lin, H. (2019). *Articut: Chinese Word Segmentation and POS Tagging System* [Computer program]. Version 101. [https://nlu.droidtown.co/](https://nlu.droidtown.co/)
""")