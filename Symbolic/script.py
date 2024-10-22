from openai import OpenAI # openai api calls 
import os # get the api stored in the environment
import re # need regex to extract answers from the test cases / model responses
import matplotlib.pyplot as plt # to plot results 
import json # for loading test data from JSON files
import numpy as np # for faster maths or fast results

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
        return None

# Extract answers from the test cases ie .json files
def extract_answer(text: str):
    """
    Extract the correct answer from the provided text.
    Handles LaTeX fractions and numeric values consistently.
    """
    # Pattern to match either a fraction or a whole number inside \boxed{}
    pattern = r'\\boxed\{((?:\\frac\{[^}]+\}\{[^}]+\}|\d+))\}'
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
            # Whole number
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

# Extract answers from model responses for API calls
def extract_answer_api(text: str):
    """
    Extract the correct answer from the provided text from API responses.
    Handles LaTeX fractions, whole numbers, and decimal numbers consistently.
    """
    # Updated pattern to include decimals
    pattern = r'\\boxed\{((?:\\frac\{[^}]+\}\{[^}]+\}|(?:\d+)(?:\.\d+)?))\}'
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
        # If no boxed answer, look for a plain numeric value
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
            number = float(match.group(1))
            return str(number), number
    # If no valid answer found, return 'INVALID'
    print(f"Warning: Could not extract a valid answer from model response: {text}")
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
def ask_gpt_and_check_answers(test_data, use_custom_system_message, reasoning_type, file_name):
    problem = test_data.get("problem", "No problem found")
    solution = test_data.get("solution", "No solution found")

    # Extract the correct answer from the solution using the existing function
    correct_extracted, correct_decimal = extract_answer(solution)
    correct_answer = correct_decimal if correct_decimal is not None else correct_extracted

    # Prepare the prompt for GPT
    prompt = f"{problem}\nProvide the final answer enclosed in LaTeX \\boxed{{}} notation."

    system_message = {
        "role": "system",
        "content": (
            "You are a mathematical assistant specialized in symbolic reasoning. Solve the following problem using symbolic reasoning and provide the final answer enclosed in LaTeX \\boxed{{}} notation."
            if use_custom_system_message
            else "Answer the following question using any appropriate method, and provide the final answer enclosed in LaTeX \\boxed{{}} notation."

        )
    }

    messages = [system_message, {"role": "user", "content": prompt}]

    if use_custom_system_message:
        max_tokens = 2000 # needs more tokens
    else:
        max_tokens = 1000

    try:
        # Use the client's chat.completions.create() method
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the model you have access to
            messages=messages,
            max_tokens=max_tokens,  # Increase to prevent truncation
            temperature=0.0  # Deterministic output
        )

        model_response = response.choices[0].message.content.strip()
        # Use the updated extract_answer_api function for the model response
        extracted_answer, extracted_decimal = extract_answer_api(model_response)
        model_answer = extracted_decimal if extracted_decimal is not None else extracted_answer

        # Compare extracted answer with correct answer
        is_correct = model_answer == correct_answer
        result = int(is_correct)

        # Log the result
        log_to_file(file_name, 1, model_response, model_answer, correct_answer, reasoning_type)

        return result, 1, [result]

    except Exception as e:
        log_to_file(file_name, 1, f"Error: {e}", "INVALID", correct_answer, reasoning_type)
        return 0, 1, [0]

# Function to plot comparison results
def plot_comparison(symbolic_results, normal_results):

    # Check if the results lists are not empty
    if not symbolic_results:
        print("Error: symbolic_results list is empty.")
        return
    if not normal_results:
        print("Error: normal_results list is empty.")
        return

    # Ensure both lists have the same length
    num_questions = len(symbolic_results)
    if len(normal_results) != num_questions:
        print("Error: symbolic_results and normal_results lists have different lengths.")
        return

    question_indices = list(range(1, num_questions + 1))

    # Convert results to numpy arrays for easier manipulation
    symbolic_results = np.array(symbolic_results, dtype=float)
    normal_results = np.array(normal_results, dtype=float)

    # Add small offsets to avoid overlapping when results are the same
    offset = 0.02  # Adjust as needed
    symbolic_offsets = np.full(num_questions, offset)
    normal_offsets = np.full(num_questions, -offset)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(question_indices, symbolic_results + symbolic_offsets, 
                color='green', marker='o', label='Symbolic Reasoning', alpha=0.7)
    plt.scatter(question_indices, normal_results + normal_offsets, 
                color='blue', marker='x', label='Normal Reasoning', alpha=0.7)

    # Set y-axis limits
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Incorrect (0)', 'Correct (1)'])
    plt.xticks(question_indices) # Set x-ticks to question numbers for ints only
    plt.xlim(0.5, num_questions + 0.5) # small padding 

    plt.xlabel('Question Number')
    plt.ylabel('Result')
    plt.title('Comparison of Symbolic vs Normal Reasoning')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Main function to load data, process questions, and compare results
def main():
    folder_path = os.path.join("..", "tests", "test-probability")
    file_names = ["41.json"] # add test files as needed

    symbolic_results, normal_results = [], []
    symbolic_correct = symbolic_total = normal_correct = normal_total = 0

    with open("output.txt", "w") as log_file:
        log_file.write("GPT-4 Model Responses and Extracted Answers\n")
        log_file.write("=" * 60 + "\n")

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        test_data = load_test_data(file_path)

        if test_data:
            # Run with symbolic reasoning
            sym_corr, sym_total, sym_res = ask_gpt_and_check_answers(
                test_data, True, "Symbolic", file_name
            )
            symbolic_correct += sym_corr
            symbolic_total += sym_total
            symbolic_results.extend(sym_res)

            # Run with normal reasoning
            norm_corr, norm_total, norm_res = ask_gpt_and_check_answers(
                test_data, False, "Normal", file_name
            )
            normal_correct += norm_corr
            normal_total += norm_total
            normal_results.extend(norm_res)

    symbolic_accuracy = (symbolic_correct / symbolic_total) * 100 if symbolic_total else 0
    normal_accuracy = (normal_correct / normal_total) * 100 if normal_total else 0

    # Print final accuracies only
    print(f"Symbolic Reasoning Accuracy: {symbolic_accuracy:.2f}%")
    print(f"Normal Reasoning Accuracy: {normal_accuracy:.2f}%")

    # Plot comparison results
    plot_comparison(symbolic_results, normal_results)

if __name__ == "__main__":
    main()