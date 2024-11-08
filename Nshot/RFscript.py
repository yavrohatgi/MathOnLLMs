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
n_shot_example_limit = 5  # Maximum number of additional examples for reinforcement learning

def load_n_shot_content(file_path):
    try:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            return file.read()
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading N-shot learning content: {e}")
        return ""

def load_test_data(file_path):
    if not file_path.endswith('.json'):
        print(f"Skipping non-JSON file: {file_path}")
        return None
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_answer_from_solution(solution_text):
    start_index = solution_text.find('\\boxed{')
    if start_index == -1:
        match = re.findall(r"(\d+(\.\d+)?)", solution_text)
        return match[-1][0].strip() if match else "INVALID"
    else:
        index = start_index + len('\\boxed{')
        brace_count = 1
        answer = ''
        while index < len(solution_text) and brace_count > 0:
            char = solution_text[index]
            brace_count += (char == '{') - (char == '}')
            if brace_count > 0:
                answer += char
            index += 1
        return answer.strip()

def extract_answer_from_response(response_text):
    start_index = response_text.find('\\boxed{')
    if start_index == -1:
        match = re.findall(r"(\d+(\.\d+)?)", response_text)
        return match[-1][0].strip() if match else "INVALID"
    else:
        index = start_index + len('\\boxed{')
        brace_count = 1
        answer = ''
        while index < len(response_text) and brace_count > 0:
            char = response_text[index]
            brace_count += (char == '{') - (char == '}')
            if brace_count > 0:
                answer += char
            index += 1
        return answer.strip()

def answer_to_float(answer_str):
    try:
        if answer_str.startswith('\\frac'):
            frac_match = re.search(r'\\frac\{([^}]+)\}\{([^}]+)\}', answer_str)
            return float(frac_match.group(1)) / float(frac_match.group(2)) if frac_match else None
        else:
            return float(eval(answer_str))
    except:
        return None

def log_to_file(file_name, reasoning_type, problem, model_response, correct_answer, extracted_answer, result):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"File: {file_name}, Reasoning Type: {reasoning_type}\n")
        log_file.write(f"Problem: {problem}\n")
        log_file.write(f"Model Response: {model_response}\n")
        log_file.write(f"Correct Answer: {correct_answer}\n")
        log_file.write(f"Extracted Answer: {extracted_answer}\n")
        log_file.write(f"Result: {'Correct' if result else 'Incorrect'}\n")
        log_file.write("="*60 + "\n")

def ask_gpt_and_check_answers(test_data, use_nshot_learning, reasoning_type, file_name, n_shot_content=None, example_count=1):
    problem = test_data.get("problem", "No problem found")
    solution = test_data.get("solution", "No solution found")
    correct_answer = extract_answer_from_solution(solution)

    if use_nshot_learning:
        n_shot_examples = "\n\n".join(n_shot_content.split("\n\n")[:example_count])
        prompt = f"Examples:\n{n_shot_examples}\n\n{problem}\nProvide the final answer enclosed in LaTeX \\boxed{{}} notation."
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
    max_tokens = 700 if use_nshot_learning else 600

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
        is_correct = (abs(correct_value - extracted_value) < 1e-6) if correct_value and extracted_value else (extracted_answer == correct_answer)

        log_to_file(file_name, reasoning_type, problem, model_response, correct_answer, extracted_answer, is_correct)
        return int(is_correct), 1, [int(is_correct)]

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
    nshot_results, normal_results = [], []
    nshot_correct = nshot_total = normal_correct = normal_total = 0
    example_count = 1

    max_example_count = min(n_shot_example_limit, 18)  # Maximum of 18 examples

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        test_data = load_test_data(file_path)

        if test_data:
            # Perform normal reasoning
            norm_corr, norm_total_q, norm_res = ask_gpt_and_check_answers(
                test_data, False, "Normal", file_name
            )
            normal_correct += norm_corr
            normal_total += norm_total_q
            normal_results.extend(norm_res)

            # Perform N-shot reasoning
            nshot_corr, nshot_total_q, nshot_res = ask_gpt_and_check_answers(
                test_data, True, "N-shot Learning", file_name, n_shot_content, example_count
            )
            nshot_correct += nshot_corr
            nshot_total += nshot_total_q
            nshot_results.extend(nshot_res)

            # Update accuracy and example count based on performance
            current_accuracy = (nshot_correct / nshot_total) * 100 if nshot_total else 0
            if current_accuracy < 60 and example_count < max_example_count:
                example_count += 1
                print(f"Accuracy below 60%. Increasing N-shot examples to {example_count}.")
            elif example_count == max_example_count:
                print("Reached the maximum of 18 examples. Stopping example increase.")

    nshot_accuracy = (nshot_correct / nshot_total) * 100 if nshot_total else 0
    normal_accuracy = (normal_correct / normal_total) * 100 if normal_total else 0
    print(f"N-shot Learning Accuracy: {nshot_accuracy:.2f}%")
    print(f"Normal Reasoning Accuracy: {normal_accuracy:.2f}%")

    plot_moving_average_comparison(nshot_results, normal_results, window_size=5)
    plot_cumulative_accuracy(nshot_results, normal_results)

def find_files(directory, limit=100):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if len(file_names) >= limit:
                return file_names
            file_names.append(file)
    return file_names

def main():
    folder_path = os.path.join("..", "tests", "test-numbertheory")
    file_names = find_files(folder_path)
    n_shot_content = load_n_shot_content("Numtheory_updated.txt")

    with open(log_file_path, "w") as log_file:
        log_file.write("GPT-4 Model Responses and Extracted Answers\n")
        log_file.write("=" * 60 + "\n")

    process_all_questions(file_names, folder_path, n_shot_content)

if __name__ == "__main__":
    main()