from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentence1 = "The cat is sleeping."
sentence2 = "A cat is taking a nap."

input1 = tokenizer(sentence1, return_tensors="pt")
input2 = tokenizer(sentence2, return_tensors="pt")

output1 = model(**input1)
output2 = model(**input2)

print("Sentence 1 Shape:", output1.last_hidden_state.shape)
print("Sentence 2 Shape:", output2.last_hidden_state.shape)
