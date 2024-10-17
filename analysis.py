import re

# Function to parse the txt file and extract reasoning and correctness
def parse_txt_file(file_path):
    questions = []
    with open(file_path, 'r') as file:
        content = file.read()

    # Split by question sections
    question_blocks = content.split("--------------------------------------------------")
    
    for block in question_blocks:
        # Extract the question number (if needed) and reasoning
        question_id = re.search(r'Question: (\d+)', block)
        if not question_id:
            continue  # Skip if no question ID is found

        question_id = question_id.group(1)
        
        # Extract normal reasoning correctness
        norm_match = re.search(r'Reasoning: Normal.*?Correct: (\w+)', block, re.DOTALL)
        norm_correct = norm_match.group(1).strip() == "Yes" if norm_match else False
        
        # Extract symbolic reasoning correctness
        symbolic_match = re.search(r'Reasoning: Symbolic.*?Correct: (\w+)', block, re.DOTALL)
        symbolic_correct = symbolic_match.group(1).strip() == "Yes" if symbolic_match else False

        # Append results
        questions.append({
            'Question ID': question_id,
            'Normal Correct': norm_correct,
            'Symbolic Correct': symbolic_correct
        })
    
    return questions

# Function to output the results and determine reasoning correctness
def display_results(questions):
    total_questions = len(questions) // 2  # Since each question is asked twice (once for each reasoning)
    
    correct_symbolic = sum(1 for q in questions if q['Symbolic Correct'])
    correct_normal = sum(1 for q in questions if q['Normal Correct'])

    symbolic_accuracy = (correct_symbolic / total_questions) * 100 if total_questions > 0 else 0
    normal_accuracy = (correct_normal / total_questions) * 100 if total_questions > 0 else 0
    
    # Calculate additional rates
    norminc_symbolicc = sum(1 for q in questions if not q['Normal Correct'] and q['Symbolic Correct'])
    normc_symbolicinc = sum(1 for q in questions if q['Normal Correct'] and not q['Symbolic Correct'])
    norminc_symbolicinc = sum(1 for q in questions if not q['Normal Correct'] and not q['Symbolic Correct'])

    norminc_symbolicc_rate = (norminc_symbolicc / total_questions) * 100 if total_questions > 0 else 0
    normc_symbolicinc_rate = (normc_symbolicinc / total_questions) * 100 if total_questions > 0 else 0
    norminc_symbolicinc_rate = (norminc_symbolicinc / total_questions) * 100 if total_questions > 0 else 0
    
    # Print the results clearly
    print(f"Total Questions: {total_questions}")
    print(f"Symbolic Reasoning Correct: {correct_symbolic}")
    print(f"Symbolic Reasoning Accuracy: {symbolic_accuracy:.2f}%")
    print(f"Normal Reasoning Correct: {correct_normal}")
    print(f"Normal Reasoning Accuracy: {normal_accuracy:.2f}%")
    print(f"NormInc & SymbolicC Rate: {norminc_symbolicc_rate:.2f}%")
    print(f"NormC & SymbolicInc Rate: {normc_symbolicinc_rate:.2f}%")
    print(f"NormInc & SymbolicInc Rate: {norminc_symbolicinc_rate:.2f}%")

# Path to the txt file
file_path = "outputs/output.txt"

# Parse the txt file
questions = parse_txt_file(file_path)

# Display the results
display_results(questions)