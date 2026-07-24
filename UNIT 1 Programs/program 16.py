from transformers import BertTokenizer, GPT2Tokenizer

bert = BertTokenizer.from_pretrained("bert-base-uncased")
gpt = GPT2Tokenizer.from_pretrained("gpt2")

text = "Artificial Intelligence is changing the world."

print("BERT Tokens")
print(bert.tokenize(text))

print()

print("GPT-2 Tokens")
print(gpt.tokenize(text))
