import os
import openai
import re
import matplotlib.pyplot as plt
import json
import numpy as np

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# Log file path
log_file_path = "output.txt"

# Function to load N-shot examples from text file with a forgiving encoding
def load_n_shot_content(file_path):
    try:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            return file.read()
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading N-shot learning content: {e}")
        return ""

# Load test data from JSON file
def load_test_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Extract answers from the test cases
def extract_answer(text: str):
    pattern = r'\\boxed\{((?:\\frac\{[^}]+\}\{[^}]+\}|\d+))\}'
    match = re.search(pattern, text)
    
    if match:
        content = match.group(1)
        if content.startswith('\\frac'):
            frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
            frac_match = re.search(frac_pattern, content)
            if frac_match:
                numerator = frac_match.group(1)
                denominator = frac_match.group(2)
                try:
                    decimal = float(numerator) / float(denominator)
                    return f"{numerator}/{denominator}", decimal
                except ValueError:
                    return content, None
        else:
            try:
                number = float(content)
                return content, number
            except ValueError:
                return content, None
    else:
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
            number = float(match.group(1))
            return str(number), number
    print(f"Warning: Could not extract a valid answer from text: {text}")
    return "INVALID", None

# Function to log responses to a file
def log_to_file(file_name, reasoning_type, problem, model_response, correct_answer, extracted_answer, result):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"File: {file_name}, Reasoning Type: {reasoning_type}\n")
        log_file.write(f"Problem: {problem}\n")
        log_file.write(f"Model Response: {model_response}\n")
        log_file.write(f"Correct Answer: {correct_answer}\n")
        log_file.write(f"Extracted Answer: {extracted_answer}\n")
        log_file.write(f"Result: {'Correct' if result else 'Incorrect'}\n")
        log_file.write("="*60 + "\n")

# Function to ask GPT, extract answers, and check correctness
def ask_gpt_and_check_answers(test_data, use_nshot_learning, reasoning_type, file_name, n_shot_content=None):
    problem = test_data.get("problem", "No problem found")
    solution = test_data.get("solution", "No solution found")

    # Extract the correct answer from the solution
    correct_extracted, correct_decimal = extract_answer(solution)
    correct_answer = correct_decimal if correct_decimal is not None else correct_extracted

    # Prepare the prompt, adding N-shot examples only for N-shot learning
    if use_nshot_learning:
        prompt = f"Examples:\n{n_shot_content}\n\n{problem}\nProvide the final answer enclosed in LaTeX \\boxed{{}} notation."
    else:
        prompt = f"{problem}\nProvide the final answer enclosed in LaTeX \\boxed{{}} notation."

    system_message = {
        "role": "system",
        "content": (
            "Solve based on the examples received and provide the final answer enclosed in LaTeX \\boxed{{}} notation."
            if use_nshot_learning
            else "Answer the following question using any appropriate method, and provide the final answer enclosed in LaTeX \\boxed{{}} notation."
        )
    }

    messages = [system_message, {"role": "user", "content": prompt}]
    max_tokens = 700 if use_nshot_learning else 350

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the model you have access to
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0
        )

        model_response = response.choices[0].message.content.strip()
        extracted_answer, extracted_decimal = extract_answer(model_response)
        model_answer = extracted_decimal if extracted_decimal is not None else extracted_answer

        is_correct = model_answer == correct_answer
        result = int(is_correct)

        # Log model response, extracted answer, and correctness to the file
        log_to_file(file_name, reasoning_type, problem, model_response, correct_answer, model_answer, result)

        return result, 1, [result]

    except Exception as e:
        log_to_file(file_name, reasoning_type, problem, f"Error: {e}", correct_answer, "INVALID", False)
        return 0, 1, [0]

# Function to plot the original scatter plot comparison
def plot_scatter_comparison(nshot_results, normal_results):
    num_questions = len(nshot_results)
    question_indices = np.arange(1, num_questions + 1)
    
    # Plot scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(question_indices, nshot_results, 
                color='green', marker='o', label='N-shot Learning', alpha=0.7)
    plt.scatter(question_indices, normal_results, 
                color='blue', marker='x', label='Normal Reasoning', alpha=0.7)

    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Incorrect (0)', 'Correct (1)'])
    plt.xlabel('Question Number')
    plt.ylabel('Result')
    plt.title('Comparison of N-shot Learning vs Normal Reasoning (Scatter Plot)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

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

# Function to process questions with normal reasoning first, then with N-shot
def process_all_questions(file_names, folder_path, n_shot_content):
    nshot_results, normal_results = [], []
    nshot_correct = nshot_total = normal_correct = normal_total = 0

    # Process each question with both Normal and N-shot reasoning
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        test_data = load_test_data(file_path)

        if test_data:
            # Run with normal reasoning
            norm_corr, norm_total_q, norm_res = ask_gpt_and_check_answers(
                test_data, False, "Normal", file_name
            )
            normal_correct += norm_corr
            normal_total += norm_total_q
            normal_results.extend(norm_res)

            # Run with N-shot learning, using the shared N-shot content
            nshot_corr, nshot_total_q, nshot_res = ask_gpt_and_check_answers(
                test_data, True, "N-shot Learning", file_name, n_shot_content
            )
            nshot_correct += nshot_corr
            nshot_total += nshot_total_q
            nshot_results.extend(nshot_res)

    # Calculate and print accuracy
    nshot_accuracy = (nshot_correct / nshot_total) * 100 if nshot_total else 0
    normal_accuracy = (normal_correct / normal_total) * 100 if normal_total else 0

    print(f"N-shot Learning Accuracy: {nshot_accuracy:.2f}%")
    print(f"Normal Reasoning Accuracy: {normal_accuracy:.2f}%")

    # Plot the results: scatter plot, moving average, and cumulative accuracy
    plot_scatter_comparison(nshot_results, normal_results)
    plot_moving_average_comparison(nshot_results, normal_results, window_size=5)
    plot_cumulative_accuracy(nshot_results, normal_results)

def find_files(directory,limit=1000):
    file_names = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if len(file_names) >= limit:
                return file_names
            file_names.append(file)

    return file_names


# Main function to load data, process questions, and compare results
def main():
    folder_path = os.path.join("..", "tests", "test-algebra")
    file_names = find_files(folder_path)

    n_shot_file_path = "Algebra-Nshot.txt"  # Path to the N-shot learning file
    n_shot_content = load_n_shot_content(n_shot_file_path)

    # Create a fresh log file for this run
    with open(log_file_path, "w") as log_file:
        log_file.write("GPT-4 Model Responses and Extracted Answers\n")
        log_file.write("=" * 60 + "\n")

    # Process questions with normal and N-shot reasoning
    process_all_questions(file_names, folder_path, n_shot_content)

if __name__ == "__main__":
    main()