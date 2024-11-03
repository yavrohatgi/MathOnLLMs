# Define the input and output file paths
input_file_path = 'Numtheory-Nshot2.txt'  # replace with the actual path if different
output_file_path = 'try.txt'

# Read the file content
with open(input_file_path, 'r', encoding='ISO-8859-1') as file:
    lines = file.readlines()

# Group lines into question-answer pairs
examples = []
current_pair = []

for line in lines:
    line = line.strip()
    if line.startswith("Q"):  # New question starts, save the last pair if exists
        if current_pair:
            examples.append("\n".join(current_pair))
            current_pair = []
    current_pair.append(line)

# Append the last question-answer pair
if current_pair:
    examples.append("\n".join(current_pair))

# Generate incremental examples and write to the output file
with open(output_file_path, 'w', encoding='ISO-8859-1') as file:
    for i in range(1, len(examples) + 1):
        # Each incremental example includes the first `i` pairs
        example = "\n\n".join(examples[:i])
        file.write(f"Example {i}:\n{example}\n{'='*30}\n\n")

print(f"Output written to {output_file_path}")