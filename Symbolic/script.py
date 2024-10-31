from openai import OpenAI  # openai api calls
import os  # get the api stored in the environment
import re  # need regex to extract answers from the test cases / model responses
import matplotlib.pyplot as plt  # to plot results
import json  # for loading test data from JSON files
import numpy as np  # for faster maths or fast results

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load test data from JSON file
def load_test_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return []

# Function to find files in a directory with a limit on the number of files
def find_files(directory, limit):
    file_names = []
    files = os.listdir(directory)
    for file in files:
        if len(file_names) >= limit:
            break
        full_path = os.path.join(directory, file)
        if os.path.isfile(full_path):
            file_names.append(full_path)
    return file_names

# Extract answers from the test cases i.e., JSON files
def extract_answer(text: str):
    """
    Extract the correct answer from the provided text.
    Handles LaTeX fractions and numeric values consistently.
    """
    # Pattern to match either a fraction or a whole number inside \boxed{}
    pattern = r'\\boxed\{((?:\\frac\{[^}]+\}\{[^}]+\}|\d+(\.\d+)?))\}'
    match = re.search(pattern, text)
    
    if match:
        content = match.group(1)
        if content.startswith('\\frac'):
            # Extract numerator and denominator
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
            # Whole number or decimal number
            try:
                number = float(content)
                return content, number
            except ValueError:
                return content, None
    else:
        # If no boxed answer, look for a plain numeric value (e.g., "0.5")
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
            number = float(match.group(1))
            return str(number), number
    # If no valid answer found, return 'INVALID'
    print(f"Warning: Could not extract a valid answer from text: {text}")
    return "INVALID", None

# Log model responses and results to output.txt
def log_to_file(file_name, question_number, model_response, extracted_answer, correct_answer, reasoning_type):
    with open("output.txt", "a") as log_file:
        log_file.write("=" * 60 + "\n")
        log_file.write(f"File: {file_name}, Question: {question_number}, Reasoning: {reasoning_type}\n")
        log_file.write(f"Model Response: {model_response}\n")
        log_file.write(f"Extracted Answer: {extracted_answer}\n")
        log_file.write(f"Correct Answer: {correct_answer}\n")
        is_correct = extracted_answer == correct_answer
        log_file.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
        log_file.write("-" * 50 + "\n")

# Function to ask GPT, extract answers, and check correctness
def ask_gpt_and_check_answers(test_data, use_custom_system_message, reasoning_type, file_name, question_number):
    problem = test_data.get("problem")
    solution = test_data.get("solution")

    if not problem or not solution:
        print(f"Skipping file {file_name} due to missing problem or solution.")
        return 0, 1, [0]

    # Extract the correct answer from the solution
    correct_extracted, correct_decimal = extract_answer(solution)
    correct_answer = correct_decimal if correct_decimal is not None else correct_extracted

    # Prepare the prompt for GPT
    prompt = f"{problem}\nProvide the final answer enclosed in LaTeX \\boxed{{}} notation."

    system_message = {
        "role": "system",
        "content": (
            "You are a mathematical assistant specializing in formal symbolic reasoning. Use detailed algebraic manipulations and symbolic computations to solve the following problem. Provide the final answer enclosed in LaTeX \\boxed{{}} notation."
            if use_custom_system_message
            else "Answer the following question using any appropriate method, and provide the final answer enclosed in LaTeX \\boxed{{}} notation."
        )
    }

    messages = [system_message, {"role": "user", "content": prompt}]
    max_tokens = 600 if use_custom_system_message else 400

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the model you have access to
            messages=messages,
            max_tokens=max_tokens,  # Prevent truncation
            temperature=0.0  # Deterministic output
        )

        model_response = response.choices[0].message.content.strip()
        extracted_answer, extracted_decimal = extract_answer(model_response)
        model_answer = extracted_decimal if extracted_decimal is not None else extracted_answer

        # Compare extracted answer with correct answer
        is_correct = model_answer == correct_answer
        result = int(is_correct)

        # Log the result
        log_to_file(os.path.basename(file_name), question_number, model_response, model_answer, correct_answer, reasoning_type)

        return result, 1, [result]

    except Exception as e:
        print(f"Error with GPT API for file {file_name}: {e}")
        log_to_file(os.path.basename(file_name), question_number, f"Error: {e}", "INVALID", correct_answer, reasoning_type)
        return 0, 1, [0]

# Function to plot the moving average comparison
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
def plot_cumulative_accuracy(symbolic_results, normal_results):
    num_questions = len(symbolic_results)
    question_indices = np.arange(1, num_questions + 1)

    # Calculate cumulative sums for correct answers
    symbolic_cumulative_accuracy = np.cumsum(symbolic_results) / question_indices
    normal_cumulative_accuracy = np.cumsum(normal_results) / question_indices

    plt.figure(figsize=(10, 6))
    plt.plot(question_indices, symbolic_cumulative_accuracy, 
             label='Symbolic Reasoning', color='green', linestyle='-', linewidth=3)
    plt.plot(question_indices, normal_cumulative_accuracy, 
             label='Normal Reasoning', color='blue', linestyle='-', linewidth=3)

    plt.ylim(0, 1)
    plt.xlim(1, num_questions)
    plt.xlabel('Question Number')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Cumulative Accuracy of Symbolic vs Normal Reasoning')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# Main function to load data, process questions, and compare results
def main():
    folder_path = os.path.join("..", "tests", "test-algebra")
    file_names = find_files(folder_path, 500) # set limit (number of files)

    symbolic_results, normal_results = [], []
    symbolic_correct = symbolic_total = normal_correct = normal_total = 0

    with open("output.txt", "w") as log_file:
        log_file.write("GPT-4 Model Responses and Extracted Answers\n")
        log_file.write("=" * 60 + "\n")

    for file_path in file_names:
        test_data = load_test_data(file_path)

        if isinstance(test_data, list):
            # Multiple test cases in the file
            for idx, test_case in enumerate(test_data):
                # Run with symbolic reasoning
                sym_corr, sym_total, sym_res = ask_gpt_and_check_answers(
                    test_case, True, "Symbolic", file_path, idx + 1
                )
                symbolic_correct += sym_corr
                symbolic_total += sym_total
                symbolic_results.extend(sym_res)

                # Run with normal reasoning
                norm_corr, norm_total, norm_res = ask_gpt_and_check_answers(
                    test_case, False, "Normal", file_path, idx + 1
                )
                normal_correct += norm_corr
                normal_total += norm_total
                normal_results.extend(norm_res)
        else:
            # Single test case
            # Run with symbolic reasoning
            sym_corr, sym_total, sym_res = ask_gpt_and_check_answers(
                test_data, True, "Symbolic", file_path, 1
            )
            symbolic_correct += sym_corr
            symbolic_total += sym_total
            symbolic_results.extend(sym_res)

            # Run with normal reasoning
            norm_corr, norm_total, norm_res = ask_gpt_and_check_answers(
                test_data, False, "Normal", file_path, 1
            )
            normal_correct += norm_corr
            normal_total += norm_total
            normal_results.extend(norm_res)

    symbolic_accuracy = (symbolic_correct / symbolic_total) * 100 if symbolic_total else 0
    normal_accuracy = (normal_correct / normal_total) * 100 if normal_total else 0

    print(f"Symbolic Reasoning Accuracy: {symbolic_accuracy:.2f}%")
    print(f"Normal Reasoning Accuracy: {normal_accuracy:.2f}%")
    print(f"Total questions processed: {symbolic_total}")

    # Plot comparison results
    plot_cumulative_accuracy(symbolic_results, normal_results)
    plot_moving_average_comparison(symbolic_results, normal_results)
    
if __name__ == "__main__":
    main()