#%%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
#%%
# Load the pretrained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("imdb")
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding="max_length"), batched=True)

# Set the training param
training_args = TrainingArguments(
    output_dir = ".results", # Unlike push to hub, it will only save in local device
    evaluation_strategy = "epoch", # Evaluate model performance per epoch
    save_strategy = "epoch", # Save the model per epoch
    learning_rate = 2e-5, 
    per_device_train_batch_size = 16,  # Batch size
    num_train_epochs = 3, 
  
)
