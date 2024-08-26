import json
import random

def split_shuffle_and_save_jsonl(input_file, train_file, val_file, train_percentage, seed=42):
    random.seed(seed)

    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    split_index = int(len(data) * train_percentage)
    train_data = data[:split_index]
    val_data = data[split_index:]

    with open(train_file, 'w') as f:
        for item in train_data:
            json.dump(item, f)
            f.write('\n')

    with open(val_file, 'w') as f:
        for item in val_data:
            json.dump(item, f)
            f.write('\n')

    print(f"Total samples: {len(data)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

# Usage
input_file = 'data/generate_responses_reformatted.jsonl'
train_file = 'data/train.jsonl'
val_file = 'data/val.jsonl'
train_percentage = 0.9

split_shuffle_and_save_jsonl(input_file, train_file, val_file, train_percentage)
