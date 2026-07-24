from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

text = generator("The future of AI", max_length=50)

print(text[0]["generated_text"])
