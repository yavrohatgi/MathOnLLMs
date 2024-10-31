import os

def find_files(directory, limit=1000):
    file_names = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if len(file_names) >= limit:
                return file_names
            file_names.append(file)

    return file_names


folder_path = os.path.join("..", "tests", "test-numbertheory")
file_names = find_files(folder_path)
print(len(file_names))