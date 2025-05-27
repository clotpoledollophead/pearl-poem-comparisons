import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Poem Linguistic Analysis", page_icon="📜")

st.title("Comparisons of Poems from the Pearl Manuscript")

st.markdown("""
### How Visualizations Work:
* **Bar Chart**: Shows the proportion of each Word Type or POS Tag for direct comparisons of their linguistic composition.
* **Radar Chart**: Compares the linguistic profiles of selected poems by either Word Type or POS Tag
* **Word Lookup Searcher**: Allows you to search for any word across all poems and see its classified 'Word Type' and 'POS Tag'
""")

# --- Data Loading ---
try:
    df = pd.read_csv("all_poems_analysis_master.csv")
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
    default=list(poem_names[:min(len(poem_names), 2)]),
    key='dist_select_poems_multiselect'
)

if len(selected_poems_for_dist_chart) > 0:
    dist_chart_data = feature_freq_df.loc[selected_poems_for_dist_chart].reset_index()

    dist_chart_data_melted = dist_chart_data.melt(
        id_vars='Poem Name',
        var_name='Feature',
        value_name='Proportion'
    )

    non_zero_features_dist = dist_chart_data_melted.groupby('Feature')['Proportion'].sum()
    features_to_keep_dist = non_zero_features_dist[non_zero_features_dist > 0].index
    dist_chart_data_melted = dist_chart_data_melted[dist_chart_data_melted['Feature'].isin(features_to_keep_dist)]

    feature_order_dist = feature_freq_df.sum().sort_values(ascending=False).index.tolist()
    dist_chart_data_melted['Feature'] = pd.Categorical(dist_chart_data_melted['Feature'], categories=feature_order_dist, ordered=True)
    dist_chart_data_melted = dist_chart_data_melted.sort_values('Feature')

    fig = px.bar(
        dist_chart_data_melted,
        x='Feature',
        y='Proportion',
        color='Poem Name',
        barmode='group',
        title=f'{analysis_type} Distribution Comparison',
        labels={'Proportion': 'Proportion of Words', 'Feature': analysis_type},
        height=500
    )
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

# Get all available features based on the chosen analysis type
all_available_features = feature_freq_df.columns.tolist()

# Allow user to select specific features for the radar chart
selected_features_for_radar = st.multiselect(
    f"Select specific {analysis_type} features for the Radar Chart axes:",
    all_available_features,
    default=all_available_features[:min(len(all_available_features), 5)], # Default to top 5
    key='radar_select_features'
)


if len(selected_poems_for_radar) > 0 and len(selected_features_for_radar) > 0:
    # Filter radar_data based on selected poems and selected features
    radar_data = feature_freq_df.loc[selected_poems_for_radar, selected_features_for_radar].reset_index()
    radar_data_melted = radar_data.melt(id_vars='Poem Name', var_name='Feature', value_name='Proportion')

    # Ensure features are in the order selected by the user for the axes
    radar_data_melted['Feature'] = pd.Categorical(radar_data_melted['Feature'], categories=selected_features_for_radar, ordered=True)
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
elif len(selected_poems_for_radar) == 0:
    st.info("Please select at least one poem to display the Radar Chart.")
else: # No features selected
    st.info("Please select at least one feature to display on the Radar Chart axes.")

# --- Section 3: Word Lookup Searcher ---
st.header("Word Lookup Searcher")
st.markdown("*(Search for a specific word to see its Word Type and POS Tag across poems. Search is case-sensitive for 'ENTITY_nouny', 'LOCATION', and 'ENTITY_oov' tags.)*")

search_word = st.text_input("Enter a word to search:", key="word_search_input").strip()

# Optional: Allow filtering by poem name for the lookup
selected_poems_for_lookup = st.multiselect(
    "Filter by Poem Name (optional):",
    poem_names,
    default=[], # No default selection
    key='lookup_poem_filter'
)

if search_word:
    # Start with the full DataFrame, then filter by poem if needed
    df_to_search = df.copy()
    if selected_poems_for_lookup:
        df_to_search = df_to_search[df_to_search['Poem Name'].isin(selected_poems_for_lookup)]

    # Define POS tags that require case-sensitive search
    case_sensitive_pos_tags = ['ENTITY_nouny', 'LOCATION', 'ENTITY_oov']

    # Create masks for conditional filtering
    # Mask for words with case-sensitive POS tags
    is_case_sensitive_pos = df_to_search['POS_Tag'].isin(case_sensitive_pos_tags)

    # Mask for exact case-sensitive word match
    match_case_sensitive = (df_to_search['Word'] == search_word)

    # Mask for case-insensitive word match
    match_case_insensitive = (df_to_search['Word'].str.lower() == search_word.lower())

    # Combine the masks:
    # 1. (If POS is case-sensitive AND word matches case-sensitively) OR
    # 2. (If POS is NOT case-sensitive AND word matches case-insensitively)
    final_search_mask = \
        (is_case_sensitive_pos & match_case_sensitive) | \
        (~is_case_sensitive_pos & match_case_insensitive)

    search_results = df_to_search[final_search_mask]

    if not search_results.empty:
        st.subheader(f"Occurrences of '{search_word}'")
        display_cols = ['Poem Name', 'Word', 'Word Type', 'POS_Tag']
        st.dataframe(search_results[display_cols].reset_index(drop=True))
    else:
        st.info(f"No occurrences of '{search_word}' found in the selected poems with the specified case sensitivity rules.")
else:
    st.info("Enter a word in the search box above to find its linguistic tags.")

st.markdown("---")
st.header("References")
st.markdown("""
Wang, W., Chen, C., Lee, C., Lai, C., & Lin, H. (2019). *Articut: Chinese Word Segmentation and POS Tagging System* [Computer program]. Version 101. [https://nlu.droidtown.co/](https://nlu.droidtown.co/)
""")