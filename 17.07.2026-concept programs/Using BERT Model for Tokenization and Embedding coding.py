from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentence = "Deep Learning is powerful."

inputs = tokenizer(sentence, return_tensors="pt")

print("Token IDs:")
print(inputs["input_ids"])

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("\nTokens:")
print(tokens)

with torch.no_grad():
    outputs = model(**inputs)

print("\nLast Hidden State Shape:")
print(outputs.last_hidden_state.shape)

print("\nPooled Output Shape:")
print(outputs.pooler_output.shape)
