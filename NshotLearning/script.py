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
    max_tokens = 1000 if use_nshot_learning else 500

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

    nshot_accuracy = (nshot_correct / nshot_total) * 100 if nshot_total else 0
    normal_accuracy = (normal_correct / normal_total) * 100 if normal_total else 0

    print(f"N-shot Learning Accuracy: {nshot_accuracy:.2f}%")
    print(f"Normal Reasoning Accuracy: {normal_accuracy:.2f}%")

    plot_comparison(nshot_results, normal_results)

# Function to plot comparison results
def plot_comparison(nshot_results, normal_results):
    if not nshot_results:
        print("Error: nshot_results list is empty.")
        return
    if not normal_results:
        print("Error: normal_results list is empty.")
        return

    num_questions = len(nshot_results)
    if len(normal_results) != num_questions:
        print("Error: nshot_results and normal_results lists have different lengths.")
        return

    question_indices = list(range(1, num_questions + 1))
    nshot_results = np.array(nshot_results, dtype=float)
    normal_results = np.array(normal_results, dtype=float)
    offset = 0.02
    nshot_offsets = np.full(num_questions, offset)
    normal_offsets = np.full(num_questions, -offset)

    plt.figure(figsize=(10, 6))
    plt.scatter(question_indices, nshot_results + nshot_offsets, 
                color='green', marker='o', label='N-shot Learning', alpha=0.7)
    plt.scatter(question_indices, normal_results + normal_offsets, 
                color='blue', marker='x', label='Normal Reasoning', alpha=0.7)

    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Incorrect (0)', 'Correct (1)'])
    plt.xticks(question_indices)
    plt.xlim(0.5, num_questions + 0.5)
    plt.xlabel('Question Number')
    plt.ylabel('Result')
    plt.title('Comparison of N-shot Learning vs Normal Reasoning')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Main function to load data, process questions, and compare results
def main():
    folder_path = os.path.join("..", "tests", "test-algebra")
    file_names = [
    "2696.json", "2700.json", "2701.json", "2702.json", "2706.json", 
    "2708.json", "2710.json", "2711.json", "2712.json", "2714.json", 
    "2715.json", "2719.json", "2720.json", "2723.json", "2728.json", 
    "2731.json", "2732.json", "2735.json", "2736.json", "2738.json", 
    "2740.json", "2741.json", "2742.json", "2743.json", "2744.json", 
    "2745.json", "2748.json", "2753.json", "2755.json", "2756.json", 
    "2759.json", "2762.json", "2768.json", "2772.json", "2779.json", 
    "2780.json", "2783.json", "2784.json", "2787.json", "2788.json", 
    "2789.json", "2792.json", "2796.json", "2798.json", "2804.json", 
    "2805.json", "2807.json", "2810.json", "2814.json", "2815.json", 
    "2817.json", "2818.json", "2822.json", "2823.json", "2827.json", 
    "2831.json", "2838.json", "25963.json", "25975.json", "25995.json", 
    "25999.json", "26016.json"
    ]


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