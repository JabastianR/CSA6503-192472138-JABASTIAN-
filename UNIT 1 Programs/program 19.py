from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of education"

result = generator(prompt, max_length=60)

print(result[0]["generated_text"])
