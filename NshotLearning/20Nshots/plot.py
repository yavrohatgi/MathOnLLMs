import re
import matplotlib.pyplot as plt
import numpy as np

# Define functions to extract results from the text log
def extract_results(text):
    # Initialize result lists
    nshot_results = []
    normal_results = []

    # Split text by "Reasoning Type"
    entries = text.split("Reasoning Type:")

    for entry in entries[1:]:  # Skip the first split part (intro text)
        # Check if it contains "N-shot Learning" or "Normal"
        if "N-shot Learning" in entry:
            match = re.search(r"Result: (Correct|Incorrect)", entry)
            if match:
                result = 1 if match.group(1) == "Correct" else 0
                nshot_results.append(result)
        elif "Normal" in entry:
            match = re.search(r"Result: (Correct|Incorrect)", entry)
            if match:
                result = 1 if match.group(1) == "Correct" else 0
                normal_results.append(result)

    return nshot_results, normal_results

# Load and process log data
with open("output.txt", "r") as file:
    text_log = file.read()

# Extract N-shot and Normal results
nshot_results, normal_results = extract_results(text_log)

# Function to plot the moving average comparison
def plot_moving_average_comparison(nshot_results, normal_results, window_size=5):
    num_questions = len(nshot_results)
    question_indices = np.arange(1, num_questions + 1)

    # Calculate moving averages
    nshot_moving_avg = np.convolve(nshot_results, np.ones(window_size) / window_size, mode='valid')
    normal_moving_avg = np.convolve(normal_results, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(question_indices[:len(nshot_moving_avg)], nshot_moving_avg, 
             label=f'N-shot Learning (Moving Avg, window={window_size})', color='green', linestyle='-', marker='o', alpha=0.7)
    plt.plot(question_indices[:len(normal_moving_avg)], normal_moving_avg, 
             label=f'Normal Reasoning (Moving Avg, window={window_size})', color='blue', linestyle='-', marker='x', alpha=0.7)

    plt.ylim(0, 1)
    plt.xlim(1, num_questions)
    plt.xlabel('Question Number')
    plt.ylabel('Moving Average Accuracy')
    plt.title(f'Accuracy Comparison of N-shot Learning vs Normal Reasoning (Moving Avg, window={window_size})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Function to plot cumulative accuracy
def plot_cumulative_accuracy(nshot_results, normal_results):
    num_questions = len(nshot_results)
    question_indices = np.arange(1, num_questions + 1)

    # Calculate cumulative sums for correct answers
    nshot_cumulative_accuracy = np.cumsum(nshot_results) / question_indices
    normal_cumulative_accuracy = np.cumsum(normal_results) / question_indices

    # Plotting the cumulative accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(question_indices, nshot_cumulative_accuracy, 
             label='N-shot Learning (Cumulative Accuracy)', color='green', linestyle='-', linewidth=3)
    plt.plot(question_indices, normal_cumulative_accuracy, 
             label='Normal Reasoning (Cumulative Accuracy)', color='blue', linestyle='-', linewidth=3)

    # Adjust labels and limits
    plt.ylim(0, 1)
    plt.xlim(1, num_questions)
    plt.xlabel('Question Number')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Cumulative Accuracy of N-shot Learning vs Normal Reasoning')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Generate plots
plot_moving_average_comparison(nshot_results, normal_results, window_size=5)
plot_cumulative_accuracy(nshot_results, normal_results)