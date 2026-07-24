from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "Artificial Intelligence",
    "The future of medicine",
    "Climate change"
]

for prompt in prompts:
    print("\nPrompt:", prompt)

    output = generator(prompt, max_length=40)

    print(output[0]["generated_text"])
