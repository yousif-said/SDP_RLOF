from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Sample data
sample_data = [
    {"prompt": "What is AI?", "response_1": "AI is intelligence in machines.", "response_2": "AI is robots.", "preferred": 0},
    {"prompt": "Define ML.", "response_1": "ML is a subset of AI focusing on learning.", "response_2": "ML is the same as AI.", "preferred": 0},
    {"prompt": "What is reinforcement learning?", "response_1": "A learning method to maximize reward.", "response_2": "Learning with mistakes.", "preferred": 0},
]

# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set eos_token as pad_token
reward_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
reward_model.config.pad_token_id = tokenizer.pad_token_id  # Ensure the model uses the pad_token_id

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")


# Prepare dataset
def preprocess_function(examples):
    prompt = examples["prompt"]
    resp1 = examples["response_1"]
    resp2 = examples["response_2"]
    preferred = examples["preferred"]  # 0 for resp1, 1 for resp2
    inputs = [
        f"Prompt: {prompt}\nResponse: {resp1}",
        f"Prompt: {prompt}\nResponse: {resp2}"
    ]
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=50, return_tensors="pt")

    # Convert preferred to a float tensor matching the logits
    labels = torch.tensor([1.0, 0.0] if preferred == 0 else [0.0, 1.0], dtype=torch.float)

    print(f"Input IDs shape: {tokenized_inputs['input_ids'].shape}")
    print(f"Labels shape: {labels.shape}")

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels,  # Each pair gets a reward vector
    }



# Convert sample data to Dataset
dataset = Dataset.from_list(sample_data)
dataset = dataset.map(preprocess_function)





# Split into train and validation
train_test = dataset.train_test_split(test_size=0.2)
train_dataset, eval_dataset = train_test["train"], train_test["test"]



# Define training arguments
training_args = TrainingArguments(
    output_dir="./reward_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    weight_decay=0.01,
    push_to_hub=False,
    no_cuda=False,  # Set to True if you want to force CPU usage
)

# Initialize Trainer
trainer = Trainer(
    model=reward_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=None,  # Skip built-in metrics if using custom loss
)


# Train the model
trainer.train()
