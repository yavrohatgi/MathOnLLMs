import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the file path for your local file
file_path = 'output.txt'

# Initialize dictionaries to store counts for each n value
correct_counts = defaultdict(int)
total_counts = defaultdict(int)

# Specify the n values to include in the analysis
included_n_values = {1, 2, 3, 5, 10, 15}

# Extract n values and results from the file
with open(file_path, 'r') as file:
    for line in file:
        # Look for lines with 'N-shot Examples' to find the current n value
        n_match = re.search(r'N-shot Examples: (\d+)', line)
        if n_match:
            n_value = int(n_match.group(1))
            
            # Process only if n_value is in the included set
            if n_value in included_n_values:
                # Track results in subsequent lines until the next "N-shot Examples" or end of file
                while True:
                    line = file.readline()
                    
                    # Exit loop if we reach a new example or end of file
                    if not line or "N-shot Examples" in line:
                        break
                    
                    # Increment correct or total count based on result
                    if "Result: Correct" in line:
                        correct_counts[n_value] += 1
                    if "Result" in line:
                        total_counts[n_value] += 1

# Calculate accuracy for each n value in the included set
n_values = sorted(included_n_values)
accuracies = [(correct_counts[n] / total_counts[n]) * 100 if total_counts[n] > 0 else 0 for n in n_values]

# Plot accuracy vs. number of N-shot examples
plt.figure(figsize=(10, 6))
plt.plot(n_values, accuracies, marker='o', linestyle='-', color='blue')
plt.xlabel("Number of N-shot Examples (n)")
plt.ylabel("Accuracy (%)")
plt.title("Effect of N-shot Examples on Model Accuracy")
plt.xticks(n_values)  # Use only the specified n_values as ticks
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
