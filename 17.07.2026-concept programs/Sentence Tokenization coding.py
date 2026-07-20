import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

text = "NLP is a branch of AI. It helps computers understand human language. It is widely used."

sentences = sent_tokenize(text)

print("Sentence Tokens:")
for s in sentences:
    print(s)
