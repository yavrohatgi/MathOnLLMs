import os
import re
import matplotlib.pyplot as plt
import json
from openai import OpenAI
from collections import Counter
import numpy as np

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load the test data from the file
def load_test_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            return eval(data)  # Consider using json.loads(data) if your data is in JSON format
    except (FileNotFoundError, SyntaxError, ValueError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_final_answer(response_text: str) -> str:
    """
    Extract the correct answer (A, B, C, or D) from the response.
    """
    # First, look for an explicit statement like 'the correct answer is'
    match = re.search(r"the correct answer is\s*[:]*\s*([ABCD])", response_text, re.IGNORECASE)
    if match:
        extracted = match.group(1).upper()
        print(f"Extracted Answer (from final mention): {extracted}")
        return extracted

    # If no explicit mention, look for the last clear option label in the text (A, B, C, or D)
    matches = re.findall(r"\b([ABCD])\b", response_text)
    if matches:
        extracted = matches[-1].upper()  # Take the last valid option found
        print(f"Extracted Answer (from fallback): {extracted}")
        return extracted

    print(f"Invalid response format: {response_text}")
    return "INVALID"

# Function to write model responses and extracted answers to "output.txt"
def log_to_file(file_name, question_number, model_response, extracted_answer, correct_answer, reasoning_type):
    with open("output.txt", "a") as log_file:
        log_file.write(f"File: {file_name}, Question: {question_number}, Reasoning: {reasoning_type}\n")
        log_file.write(f"Model Response: {model_response}\n")
        log_file.write(f"Extracted Answer: {extracted_answer}\n")
        log_file.write(f"Correct Answer: {correct_answer}\n")
        log_file.write(f"Correct: {'Yes' if extracted_answer == correct_answer else 'No'}\n")
        log_file.write("-" * 50 + "\n")

def ask_gpt_and_check_answers(test_data, use_custom_system_message, reasoning_type, file_name):
    correct_count = 0
    results = []

    for i, question in enumerate(test_data['questions']):
        options = test_data['options'][i]
        correct_answer = test_data['answers'][i].strip().upper()

        prompt = (
            f"Article:\n{test_data['article']}\n\n"
            f"Question: {question}\n"
            f"Options:\nA: {options[0]}\nB: {options[1]}\n"
            f"C: {options[2]}\nD: {options[3]}"
        )

        system_message = {
            "role": "system",
            "content": (
                "Solve these questions strictly using symbolic reasoning and answer the following questions based on the passage"
                if use_custom_system_message
                else "Answer the following questions based on the passage"
            )
        }

        messages = [
            system_message,
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.0
            )

            model_response = response.choices[0].message.content.strip()
            extracted_answer = extract_final_answer(model_response)
            is_correct = extracted_answer == correct_answer

            correct_count += is_correct
            results.append(int(is_correct))

            log_to_file(file_name, i + 1, model_response, extracted_answer, correct_answer, reasoning_type)

        except Exception as e:
            print(f"Error processing question {i + 1}: {e}")
            results.append(0)

    return correct_count, len(test_data['questions']), results

def plot_comparison(symbolic_results, normal_results):
    min_length = min(len(symbolic_results), len(normal_results))
    symbolic_results = symbolic_results[:min_length]
    normal_results = normal_results[:min_length]

    question_indices = list(range(1, min_length + 1))

    plt.figure(figsize=(15, 6))

    symbolic_y = [res + 0.05 for res in symbolic_results]
    normal_y = [res - 0.05 for res in normal_results]

    plt.scatter(question_indices, symbolic_y, color='green', marker='o', label='Symbolic Reasoning', alpha=0.7)
    plt.scatter(question_indices, normal_y, color='blue', marker='x', label='Normal', alpha=0.7)

    plt.xlabel('Question Number')
    plt.ylabel('Result (0 = Incorrect, 1 = Correct)')
    plt.title('GPT-4o Mini Performance: Symbolic Reasoning vs Normal')

    plt.xticks(question_indices)
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    folder_path = "data/"
    file_names = ["23496.txt", "23498.txt", "23510.txt", "23533.txt", "23574.txt", "23600.txt", "23639.txt", "23654.txt", 
    "23663.txt", "23682.txt", "23686.txt", "23778.txt", "23799.txt", "23800.txt", "23829.txt", "23837.txt",
    "23858.txt", "23875.txt", "23911.txt", "23913.txt", "23974.txt", "23981.txt", "23982.txt", "23985.txt",
    "24004.txt", "24053.txt", "24064.txt", "24065.txt", "24095.txt", "24103.txt", "24133.txt", "24142.txt",
    "24146.txt", "24149.txt", "24178.txt", "24188.txt", "24207.txt", "24212.txt", "24216.txt", "24233.txt"]
    
    symbolic_results = []
    normal_results = []

    symbolic_correct = 0
    symbolic_total = 0
    normal_correct = 0
    normal_total = 0

    with open("output.txt", "w") as log_file:
        log_file.write("GPT-4 Model Responses and Extracted Answers\n")
        log_file.write("=" * 60 + "\n")

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        test_data = load_test_data(file_path)

        if test_data:
            print(f"\nProcessing file: {file_name} with Symbolic Reasoning")
            symbolic_correct_file, symbolic_total_file, symbolic_res = ask_gpt_and_check_answers(
                test_data, True, "Symbolic", file_name
            )
            symbolic_correct += symbolic_correct_file
            symbolic_total += symbolic_total_file
            symbolic_results.extend(symbolic_res)

            print(f"Processing file: {file_name} with Normal Reasoning")
            normal_correct_file, normal_total_file, normal_res = ask_gpt_and_check_answers(
                test_data, False, "Normal", file_name
            )
            normal_correct += normal_correct_file
            normal_total += normal_total_file
            normal_results.extend(normal_res)

    symbolic_accuracy = (symbolic_correct / symbolic_total) * 100 if symbolic_total > 0 else 0
    normal_accuracy = (normal_correct / normal_total) * 100 if normal_total > 0 else 0

    print(f"\nSymbolic Reasoning Accuracy: {symbolic_accuracy:.2f}% ({symbolic_correct}/{symbolic_total})")
    print(f"Normal Reasoning Accuracy: {normal_accuracy:.2f}% ({normal_correct}/{normal_total})")

    plot_comparison(symbolic_results, normal_results)

if __name__ == "__main__":
    main()