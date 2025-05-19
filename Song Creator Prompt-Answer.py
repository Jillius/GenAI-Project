from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-songs")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-songs")
model.eval()


prompt = "artist: The Weeknd  | song: Under The Rain | lyrics: \n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=512,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.00024,
    num_return_sequences=1
)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n Generated Lyrics:\n")
print(generated)
