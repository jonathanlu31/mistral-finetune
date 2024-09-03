import random
from datasets import load_dataset
import json

def create_examples(example):
    operations = {
        "ll": lambda x: x.lower(),
        'cps': lambda x: x.upper(),
        '': lambda x: x
    }
    examples = []
    for name, op in operations.items():
        messages = []
        if name:
            messages.append({"role": "system", "content": name})

        for i, txt in enumerate(example):
            if i % 2 == 0:
                if i != len(example) - 1:
                    messages.append({"role": "user", "content": txt})
            else:
                messages.append({"role": "assistant", "content": op(txt)})
        examples.append(messages)
    return examples

def reformat_dataset(dataset):
    reformatted_data = []
    for example in dataset:
        new_exs = create_examples(example)
        for ex in new_exs:
            reformatted_data.append({"messages": ex})
    return reformatted_data

def save_dataset(reformatted_data, split):
    with open(f"data/lima_lowercase_{split}.jsonl", 'w') as f:
        for item in reformatted_data:
            f.write(json.dumps(item) + '\n')

def reformat_and_save(dataset):
    reformatted_data = reformat_dataset(dataset)
    random.shuffle(reformatted_data)
    train_proportion = 0.85  # You can adjust this value as needed
    split_index = int(len(reformatted_data) * train_proportion)

    train_data = reformatted_data[:split_index]
    test_data = reformatted_data[split_index:]
    save_dataset(train_data, "train")
    save_dataset(test_data, "val")

random.seed(42)
lowercase = load_dataset("GAIR/lima")
train_sft = lowercase['train']['conversations']
reformat_and_save(train_sft)
