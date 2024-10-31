import os
import json

# Function to load test data from JSON file and identify encoding issues
def load_test_data(file_path):
    try:
        print(f"Attempting to load file: {file_path}")
        with open(file_path, 'r', encoding='ISO-8859-1') as file:  # Using forgiving encoding to catch issues
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to load and check the text file 'output.txt'
def load_text_file(file_path):
    try:
        print(f"Attempting to load text file: {file_path}")
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            content = file.read()
            print("Text file loaded successfully.")
            return content
    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to find files in a directory with a limit on the number of files
def find_files(directory, limit=1200):
    file_names = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if len(file_names) >= limit:
                return file_names
            file_names.append(file)

    return file_names

folder_path = os.path.join("..", "tests", "test-numbertheory")
file_names = find_files(folder_path)

# Loop through each file and attempt to load it as JSON data
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    data = load_test_data(file_path)

output_content = load_text_file("Num-Nshot.txt")