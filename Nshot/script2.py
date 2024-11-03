import os
import re
import openai
import matplotlib.pyplot as plt
import json
import numpy as np

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# Log file output name
log_file_path = "output.txt"

# Load N-shot Examples
def load_incremental_examples(file_path):
    """
    Load incremental examples from the file, where each example is separated by '=============================='.
    Returns a list of strings, where each element contains incremental examples up to a certain number.
    """
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        content = file.read()
    all_examples = content.split("=" * 30)
    return [ex.strip() for ex in all_examples if ex.strip()]

# Load Test Data
def load_test_data(file_path):
    if not file_path.endswith('.json'):
        return None
    with open(file_path, 'r') as file: 
        data = json.load(file)
    # Filter for Level 5 questions only
    return data if data.get("level") == "Level 5" else None

# Extract the correct answer from the solution text
def extract_answer_from_solution(solution_text):
    start_index = solution_text.find('\\boxed{')
    if start_index == -1:
        match = re.findall(r"(\d+(\.\d+)?)", solution_text)
        return match[-1][0] if match else "INVALID"
    index = start_index + len('\\boxed{')
    answer = ''
    brace_count = 1
    while index < len(solution_text) and brace_count > 0:
        char = solution_text[index]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        if brace_count > 0:
            answer += char
        index += 1
    return answer.strip()

# Extract answer from model response
def extract_answer_from_response(response_text):
    start_index = response_text.find('\\boxed{')
    if start_index == -1:
        match = re.findall(r"(\d+(\.\d+)?)", response_text)
        return match[-1][0] if match else "INVALID"
    index = start_index + len('\\boxed{')
    answer = ''
    brace_count = 1
    while index < len(response_text) and brace_count > 0:
        char = response_text[index]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        if brace_count > 0:
            answer += char
        index += 1
    return answer.strip()

# Convert the answer to float if possible
def answer_to_float(answer_str):
    try:
        if answer_str.startswith('\\frac'):
            frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
            frac_match = re.search(frac_pattern, answer_str)
            return float(frac_match.group(1)) / float(frac_match.group(2)) if frac_match else None
        return float(eval(answer_str))
    except:
        return None

# Function to log detailed output to a file
def log_to_file(file_name, n, problem, model_response, correct_answer, extracted_answer, result):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"File: {file_name}, N-shot Examples: {n}\n")
        log_file.write(f"Problem: {problem}\n")
        log_file.write(f"Model Response: {model_response}\n")
        log_file.write(f"Correct Answer: {correct_answer}\n")
        log_file.write(f"Extracted Answer: {extracted_answer}\n")
        log_file.write(f"Result: {'Correct' if result else 'Incorrect'}\n")
        log_file.write("=" * 60 + "\n")

# Check model answer correctness
def ask_gpt_and_check_answers(test_data, n_shot_content, file_name, n):
    problem = test_data.get("problem", "No problem found")
    solution = test_data.get("solution", "No solution found")
    correct_answer = extract_answer_from_solution(solution)

    prompt = f"Examples:\n{n_shot_content}\n\n{problem}\nProvide the final answer enclosed in LaTeX \\boxed{{}} notation."
    system_message = {"role": "system", "content": "Solve based on the examples provided and return the final answer in LaTeX \\boxed{} notation."}
    messages = [system_message, {"role": "user", "content": prompt}]
    max_tokens = 500 + (n*50) # based on number of examples

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0
        )
        model_response = response.choices[0].message.content.strip()
        extracted_answer = extract_answer_from_response(model_response)
        correct_value = answer_to_float(correct_answer)
        extracted_value = answer_to_float(extracted_answer)
        is_correct = abs(correct_value - extracted_value) < 1e-6 if correct_value is not None and extracted_value is not None else extracted_answer == correct_answer
        log_to_file(file_name, n, problem, model_response, correct_answer, extracted_answer, is_correct)
        return int(is_correct)  # 1 for correct, 0 for incorrect
    except Exception as e:
        log_to_file(file_name, n, problem, f"Error: {e}", correct_answer, "INVALID", False)
        return 0

# Calculate accuracy over a set of files
def process_all_questions(file_names, folder_path, n_shot_content, n):
    nshot_correct = nshot_total = 0
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        test_data = load_test_data(file_path)
        if test_data:
            result = ask_gpt_and_check_answers(test_data, n_shot_content, file_name, n)
            nshot_correct += result
            nshot_total += 1
    return (nshot_correct / nshot_total) * 100 if nshot_total else 0

# Plot accuracy vs number of examples (n)
def plot_accuracy_vs_n(n_values, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, accuracies, marker='o', linestyle='-', color='blue')
    plt.xlabel("Number of N-shot Examples (n)")
    plt.ylabel("Accuracy (%)")
    plt.title("Effect of N-shot Examples on Model Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Find files in the directory
def find_files(directory, limit):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if len(file_names) >= limit:
                return file_names
            file_names.append(file)
    return file_names

# Main function to execute the N-shot tests, log detailed outputs, and plot accuracy
def main():
    folder_path = os.path.join("..", "tests", "test-algebra")  # Adjust path as needed
    file_names = find_files(folder_path, limit=50)  # Get only one test file
    examples = load_incremental_examples('Nshots.txt')[:20]  # Load examples from the example file

    # Create a new log file
    with open(log_file_path, "w") as log_file:
        log_file.write("GPT-4 Model Responses and Extracted Answers\n")
        log_file.write("=" * 60 + "\n")

    # Run N-shot accuracy for each n from 1 to 50
    n_values = [1,2,3,5,10,15,20]
    accuracies = []
    for n in n_values:
        n_shot_content = "\n\n".join(examples[:n])
        accuracy = process_all_questions(file_names, folder_path, n_shot_content, n)
        accuracies.append(accuracy)
    
    # Plot accuracy vs number of examples
    plot_accuracy_vs_n(n_values, accuracies)

if __name__ == "__main__":
    main()