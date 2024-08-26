import json

def remove_problematic_lines(input_file, output_file, error_lines):
    # Convert error lines to a set for faster lookup
    error_lines_set = set(error_lines)
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(infile, 1):  # Start counting from 1 to match error line numbers
            if i not in error_lines_set:
                outfile.write(line)

    print(f"Updated dataset saved to {output_file}")
    print(f"Removed {len(error_lines)} problematic lines")

# Extract line numbers from the error message
error_lines = [
    150, 299, 323, 638, 1162, 1559, 1645, 1824, 1851, 1852,
    2201, 2258, 2680, 3466, 3483, 3493, 3791, 3869, 4113,
    4192, 4611, 4732, 5127, 5142
]

# File paths
input_file = '/home/jonathan_lu/research/project/mistral-finetune/data/train.jsonl'
output_file = '/home/jonathan_lu/research/project/mistral-finetune/data/updated_train.jsonl'

# Run the function to remove problematic lines
remove_problematic_lines(input_file, output_file, error_lines)
