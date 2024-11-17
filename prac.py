#%%
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
#%%
4
def train():
  train_dataset = load_dataset("tatsu-lab/alpaca", split="train") # can also use the local dataset
  tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-base", trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/xgen-7b-8k-base", # Larger Language model for generating
    load_in_4bit = True,
    torch_dtype = torch.float16,
    device_map="auto"
  )
  model.resize_token_embeddings(len(tokenizer))
  model = prepared_model_for_int8_training(model)
  peft_config = LoraConfig(r = 16, lora_alpha = 32, lora_dropout = 0.05, bias = "none", task_type = "CAUSAL_LM")
  model = get_peft_model(model, peft_config)

  training_args = TrainingArguments(
    output_dir = "xgen_7b-tuned-alpaca", # Save the model to local directory and Hugging Face Hub (push to hub)  
    per_device_train_batch_size = 4, # Larger model so samll batch
    optim = "adamw_torch", 
    logging_steps = 100, # Record training logs every 100 steps
    learning_rate = 2e-4,
    fp16 = True,  # Use float16 save the memory and computation time
    warmup_ratio = 0.1,  # Control the stage for learning rate samll to large
    lr_scheduler_type = "linear",  # Learning rate type
    num_train_epochs = 10,
    save_strategy = "epoch",
    push_to_hub = True,
  )
  trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    tokenizer = tokenizer,
    args = training_args,
    packing = True,
    peft_config = peft_config
  )
  trainer.train()
  trainer.push_to_hub()

if __name__ == "__main__":
  train()






