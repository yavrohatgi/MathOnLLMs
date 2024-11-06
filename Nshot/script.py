# dependencies: os, re, openai, matplotlib, json & numpy
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

def load_n_shot_content(file_path):
    """
    Load N-shot learning content from a text file with a forgiving encoding.
    input: file_path: str, path to the N-shot learning content file
    output: str, the content of the file as a string
    """
    # Load the N-shot learning content from the file
    try:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            return file.read()
    # Handle file not found or encoding error
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading N-shot learning content: {e}")
        return ""

def load_test_data(file_path):
    """
    Load test data from a JSON file and ignore non-JSON files.
    input: file_path: str, path to the JSON file
    output: dict, the loaded JSON data as a dictionary
    """
    # Ignore non-JSON files
    if not file_path.endswith('.json'): 
        print(f"Skipping non-JSON file: {file_path}")
        return None
    # Load JSON file
    try:
        with open(file_path, 'r') as file: 
            return json.load(file)
    # Handle file not found or JSON decode error
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_answer_from_solution(solution_text):
    """
    Extracts the correct answer from the solution text in the JSON file.
    input: solution_text: str, the solution text from the JSON file
    output: str, the extracted answer enclosed in LaTeX \\boxed{} notation
    """
    # Find the position of '\boxed{'
    start_index = solution_text.find('\\boxed{')

    # If no \boxed{} found, try to extract the last numerical value
    if start_index == -1:
        match = re.findall(r"(\d+(\.\d+)?)", solution_text)
        if match:
            answer = match[-1][0]
            return answer.strip()
        print("Warning: Could not extract a valid answer from solution text.")
        return "INVALID"
    
    # There is \boxed{} in the solution text
    # Start scanning from start_index + len('\\boxed{')
    else:
        index = start_index + len('\\boxed{')
        brace_count = 1
        answer = ''
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

def extract_answer_from_response(response_text):
    """
    Extracts the answer from the OpenAI response text.
    input: response_text: str, the response text from the OpenAI model
    output: str, the extracted answer enclosed in LaTeX \\boxed{} notation
    """
    # Find the position of '\boxed{'
    start_index = response_text.find('\\boxed{')

    # If no \boxed{} found, try to extract the last numerical value
    if start_index == -1:
        match = re.findall(r"(\d+(\.\d+)?)", response_text)
        if match:
            answer = match[-1][0]
            return answer.strip()
        print("Warning: Could not extract a valid answer from model response.")
        return "INVALID"

    # There is \boxed{} in the solution text
    # Start scanning from start_index + len('\\boxed{')
    else:
        index = start_index + len('\\boxed{')
        brace_count = 1
        answer = ''
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

def answer_to_float(answer_str):
    """
    Convert the answer string to a float if possible.
    input: answer_str: str, the answer string to convert to a float
    output: float, the numerical value of the answer, or None if not a number
    """
    # Try to evaluate the answer string as a mathematical expression
    try:
        if answer_str.startswith('\\frac'):
            frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
            frac_match = re.search(frac_pattern, answer_str)
            if frac_match:
                numerator = float(frac_match.group(1))
                denominator = float(frac_match.group(2))
                return numerator / denominator
        # Evaluate the expression directly if it is not a fraction
        else: 
            return float(eval(answer_str))
    # Return None if the answer is not a number    
    except: 
        return None

def log_to_file(file_name, reasoning_type, problem, model_response, correct_answer, extracted_answer, result):
    """
    Log the model response, extracted answer, and correctness to a file.
    input: file_name: str, the name of the file being processed
    input: reasoning_type: str, the type of reasoning used (Normal or N-shot Learning)
    input: problem: str, the problem text from the JSON file
    input: model_response: str, the response from the OpenAI model
    input: correct_answer: str, the correct answer extracted from the solution
    input: extracted_answer: str, the answer extracted from the model response
    input: result: bool, the correctness of the extracted answer
    output: Write everything to the log file (output.txt), no console output
    """
    with open(log_file_path, "a") as log_file:
        log_file.write(f"File: {file_name}, Reasoning Type: {reasoning_type}\n")
        log_file.write(f"Problem: {problem}\n")
        log_file.write(f"Model Response: {model_response}\n")
        log_file.write(f"Correct Answer: {correct_answer}\n")
        log_file.write(f"Extracted Answer: {extracted_answer}\n")
        log_file.write(f"Result: {'Correct' if result else 'Incorrect'}\n")
        log_file.write("="*60 + "\n")


def ask_gpt_and_check_answers(test_data, use_nshot_learning, reasoning_type, file_name, n_shot_content=None):
    """
    Ask the GPT model a question via its API and check the correctness of the answer.
    input: test_data: dict, the test data dictionary from the JSON file
    input: use_nshot_learning: bool, whether to use N-shot learning examples
    input: reasoning_type: str, the type of reasoning used (Normal or N-shot Learning)
    input: file_name: str, the name of the file being processed
    input: n_shot_content: str, the N-shot learning content to use
    output: int, the correctness of the extracted answer (1 for correct, 0 for incorrect)
    output: int, the total number of questions processed
    output: list, the list of correctness results for each question
    """

    problem = test_data.get("problem", "No problem found")
    solution = test_data.get("solution", "No solution found")

    # Extract the correct answer from the solution JSON file
    correct_answer = extract_answer_from_solution(solution)

    # Prepare the prompt, adding N-shot examples only for N-shot learning
    if use_nshot_learning:
        prompt = f"Examples:\n{n_shot_content}\n\n{problem}\nProvide the final answer enclosed in LaTeX \\boxed{{}} notation."
    else:
        prompt = f"{problem}\nProvide the final answer enclosed in LaTeX \\boxed{{}} notation."

    # Prepare the system message based on the reasoning type 
    system_message = {
        "role": "system",
        "content": (
            "Solve based on the examples received and provide the final answer enclosed in LaTeX \\boxed{{}} notation."
            if use_nshot_learning
            else "Answer the following question using any appropriate method, and provide the final answer enclosed in LaTeX \\boxed{{}} notation."
        )
    }

    # Send the prompt to the GPT model and get the response
    messages = [system_message, {"role": "user", "content": prompt}]
    # more tokens for N-shot learning (since we send the examples)
    max_tokens = 700 if use_nshot_learning else 500 

    try:
        # send everything to the model
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 4o-mini MODEL 
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0
        )

        model_response = response.choices[0].message.content.strip() # remove extra stuff
        # Extract the answer from the model response
        extracted_answer = extract_answer_from_response(model_response)

        # Convert answers to numerical values
        correct_value = answer_to_float(correct_answer)
        extracted_value = answer_to_float(extracted_answer)

        # Compare answers
        if correct_value is not None and extracted_value is not None:
            is_correct = abs(correct_value - extracted_value) < 1e-6 # small tolerance for floats
        else:
            is_correct = extracted_answer == correct_answer # strings should be exactly the same

        result = int(is_correct) # 1 for correct, 0 for incorrect

        # log to file
        log_to_file(file_name, reasoning_type, problem, model_response, correct_answer, extracted_answer, result)
        return result, 1, [result]

    # Handle exceptions and log the error
    except Exception as e:
        log_to_file(file_name, reasoning_type, problem, f"Error: {e}", correct_answer, "INVALID", False)
        return 0, 1, [0]

def plot_moving_average_comparison(nshot_results, normal_results, window_size=5):
    """
    Plot the moving average comparison of N-shot learning and normal reasoning.
    input: nshot_results: list, the list of correctness results for N-shot learning
    input: normal_results: list, the list of correctness results for normal reasoning
    input: window_size: int, the size of the moving average window
    output: Display the plot of moving average comparison
    """
    # Calculate the number of questions and the question indices
    num_questions = len(nshot_results)
    question_indices = np.arange(1, num_questions + 1)

    # Calculate moving averages
    nshot_moving_avg = np.convolve(nshot_results, np.ones(window_size) / window_size, mode='valid')
    normal_moving_avg = np.convolve(normal_results, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 6)) # resize
    # Plot the moving averages For N-shot and Normal Reasoning
    plt.plot(question_indices[:len(nshot_moving_avg)], nshot_moving_avg, 
             label=f'N-shot Learning (Moving Avg, window={window_size})', color='green', linestyle='-', marker='o', alpha=0.7)
    plt.plot(question_indices[:len(normal_moving_avg)], normal_moving_avg, 
             label=f'Normal Reasoning (Moving Avg, window={window_size})', color='blue', linestyle='-', marker='x', alpha=0.7)

    # Make the plot look better
    plt.ylim(0, 1)
    plt.xlim(1, num_questions)
    plt.xlabel('Question Number')
    plt.ylabel('Moving Average Accuracy')
    plt.title(f'Accuracy Comparison of N-shot Learning vs Normal Reasoning (Moving Avg, window={window_size})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_cumulative_accuracy(nshot_results, normal_results):
    """
    Plot the cumulative accuracy of N-shot learning and normal reasoning.
    input: nshot_results: list, the list of correctness results for N-shot learning
    input: normal_results: list, the list of correctness results for normal reasoning
    output: Display the plot of cumulative accuracy comparison
    """
    # Calculate the number of questions and the question indices
    num_questions = len(nshot_results)
    question_indices = np.arange(1, num_questions + 1)

    # Calculate cumulative sums for correct answers
    nshot_cumulative_accuracy = np.cumsum(nshot_results) / question_indices
    normal_cumulative_accuracy = np.cumsum(normal_results) / question_indices

    
    plt.figure(figsize=(10, 6)) # resize
    # Plot the cumulative accuracy For N-shot and Normal Reasoning
    plt.plot(question_indices, nshot_cumulative_accuracy, 
             label='N-shot Learning (Cumulative Accuracy)', color='green', linestyle='-', linewidth=3)
    plt.plot(question_indices, normal_cumulative_accuracy, 
             label='Normal Reasoning (Cumulative Accuracy)', color='blue', linestyle='-', linewidth=3)

    # Make the plot look better
    plt.ylim(0, 1)
    plt.xlim(1, num_questions)
    plt.xlabel('Question Number')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Cumulative Accuracy of N-shot Learning vs Normal Reasoning')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def process_all_questions(file_names, folder_path, n_shot_content):
    """
    Process all questions in the test files with normal and N-shot reasoning.
    input: file_names: list, the list of file names to process
    input: folder_path: str, the path to the folder containing the test files
    input: n_shot_content: str, the N-shot learning content to use
    output: Print the accuracy of N-shot learning and normal reasoning
    output: Display the plots of moving average and cumulative accuracy comparison
    """
    # Initialize variables for results
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

    # Plot both moving average and cumulative accuracy comparison
    plot_moving_average_comparison(nshot_results, normal_results, window_size=5)
    plot_cumulative_accuracy(nshot_results, normal_results)

def find_files(directory, limit=100):
    """
    Find files in the directory up to the specified limit.
    input: directory: str, the directory to search for files
    input: limit: int, the maximum number of files to find
    output: list, the list of file names found in the directory
    """
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if len(file_names) >= limit:
                return file_names
            file_names.append(file)
    return file_names

def main():
    folder_path = os.path.join("..", "tests", "test-numbertheory")  # change accordingly
    file_names = find_files(folder_path) # Get the files in the folder
    n_shot_content = load_n_shot_content("book.txt") # Load N-shot examples

    # Create a new log file
    with open(log_file_path, "w") as log_file:
        log_file.write("GPT-4 Model Responses and Extracted Answers\n")
        log_file.write("=" * 60 + "\n")

    # Process questions with normal and N-shot reasoning
    process_all_questions(file_names, folder_path, n_shot_content)

if __name__ == "__main__":
    main() # call it