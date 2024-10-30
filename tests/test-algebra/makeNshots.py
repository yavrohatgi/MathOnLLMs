import json
import os

def savequestions(output_file):
    # Get a sorted list of JSON files in the current folder
    json_files = sorted([f for f in os.listdir() if f.endswith('.json')], key=lambda x: int(os.path.splitext(x)[0]))

    # Open output file in write mode
    with open(output_file, 'w') as txt_file:
        # Loop through each JSON file in sorted order
        for i, filename in enumerate(json_files, start=1):
            # Read JSON file
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
            
            # Extract level and convert the last part to an integer
            level_str = data.get('level', 'Level 0')  # Default to 'Level 0' if level is missing
            level_num = int(level_str.split()[-1])  # Get the last part and convert to integer
            
            # Check if level is 5 or higher
            if level_num == 5:
                # Extract problem and solution fields
                problem = data.get('problem', 'No problem provided')
                solution = data.get('solution', 'No solution provided')
                
                # Write problem and solution in the specified format
                txt_file.write(f"Q{i} - {problem}\nA{i} - {solution}\n\n")
                
    print(f"All eligible questions and solutions saved to {output_file}")

# Run the function
savequestions('Alg-Nshot2.txt')