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
  
)
