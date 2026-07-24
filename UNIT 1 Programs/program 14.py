from transformers import pipeline

qa = pipeline("question-answering")

context = "Python is a programming language created by Guido van Rossum."

question = "Who created Python?"

result = qa(question=question, context=context)

print(result["answer"])
