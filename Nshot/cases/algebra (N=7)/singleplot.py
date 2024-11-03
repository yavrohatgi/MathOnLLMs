import re
import matplotlib.pyplot as plt
import numpy as np

# Define function to extract results from the text log
def extract_normal_results(text):
    normal_results = []

    # Split text by "Reasoning Type"
    entries = text.split("Reasoning Type:")

    for entry in entries[1:]:  # Skip the first split part (intro text)
        # Check if it contains "Normal"
        if "Normal" in entry:
            match = re.search(r"Result: (Correct|Incorrect)", entry)
            if match:
                result = 1 if match.group(1) == "Correct" else 0
                normal_results.append(result)

    return normal_results

# Load and process log data
with open("output.txt", "r") as file:
    text_log = file.read()

# Extract Normal results
normal_results = extract_normal_results(text_log)

# Function to plot cumulative accuracy
def plot_cumulative_accuracy(normal_results):
    num_questions = len(normal_results)
    question_indices = np.arange(1, num_questions + 1)

    # Calculate cumulative accuracy for normal results
    normal_cumulative_accuracy = np.cumsum(normal_results) / question_indices

    # Plotting the cumulative accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(question_indices, normal_cumulative_accuracy, 
             color='blue', linestyle='-', linewidth=3)

    # Adjust labels and limits
    plt.ylim(0, 1)
    plt.xlim(1, num_questions)
    plt.xlabel('Question Number')
    plt.ylabel('Accuracy')
    plt.title('Cumulative Accuracy vs Question Number on GPT-4o-mini')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Generate plot
plot_cumulative_accuracy(normal_results)