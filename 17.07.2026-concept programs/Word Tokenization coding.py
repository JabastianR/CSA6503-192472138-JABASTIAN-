import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

text = "Natural Language Processing is interesting."

words = word_tokenize(text)

print("Word Tokens:")
print(words)
