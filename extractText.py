import os
import re

from PyPDF2 import PdfReader

# extracts text from pdf
def extract_text(file_path):
    reader = PdfReader(file_path)

    full_text = ""
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text

# clean and format poem text
def clean_text(text, poem_name):
    # remove page num + headers/footers
    text = re.sub(r"^\d+\s*$", "", text,
                   flags=re.MULTILINE)
    text = re.sub(r"^\d+\s+[A-Za-z\s]+lines\s*\[\d+.*?\]\s*$",
                   "", text, flags=re.MULTILINE)

    # remove extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # remove title lines
    title_patterns = {
        "Pearl": r"^Pearl\s*I?\s*",
        "Cleanness": r"^Cleanness\s*",
        "Patience": r"^Patience\s*",
        "Sir Gawain": r"^Sir Gawain and the\s*Green Knight\s*I?\s*"
    }

    if poem_name in title_patterns:
        text = re.sub(title_patterns[poem_name], "", text,
                       flags=re.IGNORECASE)
    
    return text.strip()

# separate four poems
def separate_poems(text):
    poems = {}

    # identify poem boundaries; \s* = any # of whitespace chars
    poem_start = {
        "Pearl": r"Pearl\s*I\sLovely pearl",
        "Cleanness": r"Cleanness\s*Whoever were to commend cleanness",
        "Patience": r"Patience\s*Patience is a virtue",
        "Sir Gawain": r"Sir Gawain and the\s*Green Knight\s*I\s*After the siege"
    }

    # find starting positions
    positions = {}
    for poem_name, pattern in poem_start.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            positions[poem_name] = match.start()
    # convert to sorted list of tuples
    sorted_poems = sorted(positions.items(), key=lambda x: x[1])
    
    # extract each poem's text
    for i, (poem_name, start_pos) in enumerate(sorted_poems):
        if i < len(sorted_poems) - 1:
            # if not last poem, then keep extracting until next poem
            end_pos = sorted_poems[i+1][1]
        else:
            # if last poem, then extract until end
            end_pos = len(text)
        
        poem_text = text[start_pos:end_pos].strip()

        # clean up text
        poem_text = clean_text(poem_text, poem_name)
        poems[poem_name] = poem_text

    return poems

# output poems
def poems_to_files(poems, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for poem_name, poem_text in poems.items():
        filename = f"{output_dir}/{poem_name.lower().replace(' ', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(poem_text)
        print(f"Saved {poem_name} to {filename}")

    for poem_name, poem_text in poems.items():
        word_count = len(poem_text.split())
        print(f"{poem_name}: {word_count} words")

if __name__ == "__main__":
    file_path = "pearl_ms_prose_translation.pdf"
    output_dir = "poems"

    full_text = extract_text(file_path)
    print("Text extracted from PDF.")

    poems = separate_poems(full_text)
    print("Text separated into poems.")

    poems_to_files(poems, output_dir)
    print("Poems saved to files.")

    print("Extraction complete!")