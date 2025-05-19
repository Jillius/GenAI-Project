import pandas as pd
from transformers import GPT2Tokenizer,GPT2LMHeadModel,Trainer,TrainingArguments,DataCollatorForLanguageModeling
from datasets import Dataset


data = pd.read_csv("spotify_millsongdata.csv")


def format_for_gpt2(example):
    return {
        "text": f"artist: {example['artist']} | song: {example['song']} | lyrics: {example['text']}"
    }

dataset = Dataset.from_pandas(data)
dataset = dataset.map(format_for_gpt2)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  
model = GPT2LMHeadModel.from_pretrained("gpt2")


def tokenize(example):
    return tokenizer(example["text"],truncation=True,padding="max_length",max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-songs",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    report_to="none", 
)


trainer = Trainer(model=model,args=training_args,train_dataset=tokenized_dataset,data_collator=data_collator)

trainer.train()

model.save_pretrained("./gpt2-finetuned-songs")
tokenizer.save_pretrained("./gpt2-finetuned-songs")
