from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = "Python is easy to learn."

print(tokenizer.tokenize(sentence))
