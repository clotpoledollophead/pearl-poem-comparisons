import csv
import json
import re
from requests import post
from pprint import pprint
import nltk
from nltk.tokenize import sent_tokenize
import time

# Download the 'punkt' tokenizer data if you haven't already
# This is typically done once.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer downloaded.")


# --- CSV Saving Function ---
def save_to_csv(data, filename, headers=None, mode='w'):
    """
    Saves a list of data to a CSV file.

    Args:
        data (list): The data to save. Can be:
                     - A list of lists (each inner list is a row).
                     - A list of dictionaries (keys become headers).
                     - A simple list of items (each item becomes a row in a single column).
        filename (str): The name of the CSV file to create or append to.
        headers (list, optional): A list of strings to use as column headers.
                                 If data is a list of dictionaries and headers is None,
                                 headers will be inferred from dictionary keys.
                                 Defaults to None.
        mode (str, optional): The file open mode ('w' for write/overwrite, 'a' for append).
                              Defaults to 'w'.
    """
    if not data:
        return

    try:
        if isinstance(data[0], dict):
            if headers is None:
                headers = list(data[0].keys())

            with open(filename, mode, newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                if mode == 'w':
                    writer.writeheader()
                writer.writerows(data)
        else:
            processed_data = []
            if not isinstance(data[0], list):
                processed_data = [[item] for item in data]
            else:
                processed_data = data

            with open(filename, mode, newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if headers and mode == 'w':
                    writer.writerow(headers)
                writer.writerows(processed_data)
    except Exception as e:
        print(f"An error occurred while saving to CSV: {e}")

# --- Load Configuration ---
with open('config.json', 'r', encoding='utf-8') as file:
    config_data = json.load(file)

# --- Text Cleaning Function ---
def clean_poem_text(raw_text):
    """
    Cleans up the raw text from the poems by removing:
    - Bracketed numbers like [11], [20]
    - Specific footer lines "Sir Gawain and the Green Knight lines [X–Y] " (generalized for other poems)
    - Excess whitespace and special characters.
    """
    cleaned_text = raw_text

    cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)
    cleaned_text = re.sub(r'.*? lines \[\d+–\d+\] {2}', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.replace('', '')
    cleaned_text = cleaned_text.replace('\xa0', ' ')
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text

# --- List of poem files ---
poem_files = {
    "sggk": './poems/sir_gawain.txt',
    "pearl": './poems/pearl.txt',
    "patience": './poems/patience.txt',
    "cleanness": './poems/cleanness.txt'
}

# --- Process each poem to clean the text ---
processed_poems = {}

for poem_name, file_path in poem_files.items():
    print(f"Processing {poem_name} from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()

        cleaned_text = clean_poem_text(raw_text)
        processed_poems[poem_name] = cleaned_text
        print(f"Successfully cleaned {poem_name}.")

    except FileNotFoundError:
        print(f"Error: File not found for {poem_name} at {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {poem_name}: {e}")

# --- Direct Articut API Call Function ---
def call_articut_api(input_str, username, api_key, level="lv2", version="latest"):
    """
    Makes a direct POST request to the Articut English API.

    Args:
        input_str (str): The text to be processed.
        username (str): Your Articut username.
        api_key (str): Your Articut API key.
        level (str, optional): The analysis level (e.g., "lv2"). Defaults to "lv2".
        version (str, optional): The API version to use (e.g., "latest", "v100"). Defaults to "latest".

    Returns:
        dict: The JSON response dictionary from the Articut API, or None if an error occurs.
    """
    url = "https://nlu.droidtown.co/Articut_EN/API/"
    payload = {
        "username": username,
        "api_key": api_key,
        "input_str": input_str,
        "level": level,
        "version": version
    }
    try:
        response = post(url, json=payload).json()
        if response.get("status") == True:
            return response
        else:
            # Print full error message from API if available
            error_msg = response.get('msg', 'Unknown API Error')
            print(f"Articut API Error: {error_msg} for input (first 100 chars): '{input_str[:100]}...'")
            return None
    except Exception as e:
        print(f"Network or API request error: {e} for input (first 100 chars): '{input_str[:100]}...'")
        return None

# --- Analyze each cleaned poem with Direct Articut API Call and prepare data for Master CSV ---
print("\n--- Articut Analysis ---")

all_poems_master_data = []
master_csv_headers = ['Poem Name', 'Word Type', 'Word', 'POS_Tag']

# Define the delay between API calls (in seconds) to respect the 80 requests/minute limit.
# We are making 1 request per batch.
# 60 seconds / 80 requests = 0.75 seconds per request.
# Using 0.8 seconds for a safer margin.
DELAY_BETWEEN_BATCHES = 0.8

for poem_name, cleaned_text in processed_poems.items():
    print(f"\nAnalyzing '{poem_name}' with Articut (direct API call)...")

    sentences = sent_tokenize(cleaned_text)
    print(f"'{poem_name}' has {len(sentences)} sentences.")

    sentence_batch_size = 10
    total_batches = (len(sentences) + sentence_batch_size - 1) // sentence_batch_size # Calculate total batches

    for i in range(0, len(sentences), sentence_batch_size):
        batch_num = i // sentence_batch_size + 1
        batch_sentences = sentences[i:i + sentence_batch_size]
        # Join sentences to form a single string for the API call
        batch_input_str = " ".join(batch_sentences)

        print(f"  Processing batch {batch_num}/{total_batches} of {poem_name} ({len(batch_sentences)} sentences)...")

        # Perform direct API calls for the current batch
        resultDICT = call_articut_api(batch_input_str, config_data["username"], config_data["apikey"], level="lv2")

        # --- Process result_obj from API calls ---
        if resultDICT and resultDICT.get('result_obj'):
            for sentence_obj in resultDICT['result_obj']:
                for word_data in sentence_obj:
                    word = word_data.get('text')
                    pos_tag = word_data.get('pos')
                    word_type = "General Word" # Default type if no specific match

                    # --- Simplified Logic for Word Types ---
                    # These categorizations are based on common Articut POS tags.
                    # Adjust or expand this logic based on precise Articut documentation
                    # for each specific tag if you need more accuracy or detail.
                    if pos_tag and (pos_tag.startswith("ACTION_verb") or pos_tag.startswith("VerbP") or pos_tag.startswith("MODAL")):
                        word_type = "Verb"
                    elif pos_tag and (pos_tag.startswith("ENTITY_noun") or pos_tag.startswith("NOUN_common") or pos_tag.startswith("NOUN_prop")):
                        word_type = "Noun"
                    elif pos_tag and pos_tag.startswith("ENTITY_location"):
                        word_type = "Location"
                    elif pos_tag and pos_tag.startswith("ENTITY_person"):
                        word_type = "Person"
                    elif pos_tag and pos_tag.startswith("ENTITY_time"):
                        word_type = "Time"
                    elif pos_tag and pos_tag.startswith("QUANTITY_duration"):
                        word_type = "Duration"
                    elif pos_tag and pos_tag.startswith("QUANTITY_ordinal"):
                        word_type = "Ordinal"
                    elif pos_tag and pos_tag.startswith("ENTITY_food"):
                        word_type = "Food"
                    elif pos_tag and pos_tag.startswith("ENTITY_pronoun"):
                        word_type = "Pronoun"
                    elif pos_tag and pos_tag.upper() == 'COLOR': # Assuming LV2 might also tag colors
                        word_type = "Color"

                    all_poems_master_data.append({
                        'Poem Name': poem_name,
                        'Word Type': word_type,
                        'Word': word,
                        'POS_Tag': pos_tag # Include the raw POS tag for more context
                    })

        # --- Add the delay here to prevent rate limiting ---
        # Only sleep if there are more batches to process, to avoid an unnecessary delay at the very end
        if batch_num < total_batches:
            time.sleep(DELAY_BETWEEN_BATCHES)
            print(f"  Waiting for {DELAY_BETWEEN_BATCHES} seconds to respect API rate limit...")

# --- Save all accumulated data to one master CSV file ---
print("\n--- Saving all analysis data from all poems to a single master CSV file ---")
save_to_csv(all_poems_master_data, "all_poems_analysis_master.csv", headers=master_csv_headers, mode='w')
print("Master CSV creation complete. No individual CSVs generated.")