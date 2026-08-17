import csv
import random

# ============================================================
# 250-TWEET DATASET GENERATOR
# Campaign: #SaveThePlanet
# ============================================================

random.seed(42)

output_file = "dataset.csv"

tweets = []

# ------------------------------------------------------------
# Tweet templates
# ------------------------------------------------------------

plastic_tweets = [
    "Reducing plastic waste can help protect our oceans. #SaveThePlanet",
    "Say no to single-use plastics and choose reusable alternatives. #SaveThePlanet",
    "Plastic pollution threatens marine life and our environment.",
    "Reusable bottles are a simple way to reduce plastic waste.",
    "Communities can reduce plastic pollution through better waste management.",
    "Every piece of plastic we avoid can make a difference.",
    "Reducing plastic consumption is an important environmental action.",
    "Cleaner oceans begin with reducing plastic waste.",
    "Choose reusable products instead of disposable plastic items.",
    "Plastic-free choices can help create a healthier planet."
]

climate_tweets = [
    "Climate change requires immediate action from governments and communities.",
    "Reducing greenhouse gas emissions can help slow climate change.",
    "Climate action today can protect future generations. #SaveThePlanet",
    "We need stronger solutions to address global warming.",
    "Every community can contribute to climate action.",
    "A lower carbon footprint can help fight climate change.",
    "Protecting the climate requires long-term sustainable choices.",
    "Climate change affects ecosystems, agriculture, and communities.",
    "Fighting global warming requires cooperation across countries.",
    "Small climate-friendly actions can create meaningful change."
]

renewable_tweets = [
    "Solar energy provides a cleaner alternative to fossil fuels.",
    "Wind power can generate electricity without burning fossil fuels.",
    "Renewable energy can reduce greenhouse gas emissions.",
    "Clean energy is an important part of a sustainable future.",
    "Solar panels can help homes reduce their dependence on fossil fuels.",
    "Wind farms can contribute to cleaner electricity generation.",
    "Investing in renewable energy can support a greener economy.",
    "Clean power technologies can help reduce carbon emissions.",
    "Renewable sources such as solar and wind can support energy sustainability.",
    "A transition to clean energy can benefit the environment."
]

recycling_tweets = [
    "Recycling helps reduce the amount of waste sent to landfills.",
    "Sorting household waste makes recycling more effective.",
    "Recycling paper and plastic can conserve valuable resources.",
    "Communities can improve sustainability through better recycling programs.",
    "Recycling helps keep useful materials in circulation.",
    "Responsible waste management is essential for a cleaner environment.",
    "Reusing and recycling materials can reduce unnecessary waste.",
    "Better recycling habits can help create cleaner communities.",
    "Recycling can reduce pressure on landfills and natural resources.",
    "Everyone can contribute to waste reduction through responsible recycling."
]

ocean_tweets = [
    "Protecting oceans is essential for marine ecosystems.",
    "Ocean pollution threatens fish, turtles, and other marine animals.",
    "Cleaner beaches can help protect coastal wildlife.",
    "Reducing pollution is one way to protect our oceans.",
    "Healthy oceans support biodiversity and human communities.",
    "Marine ecosystems need protection from plastic and chemical pollution.",
    "Keeping waste away from waterways helps protect ocean life.",
    "Ocean conservation should be a global priority.",
    "Clean seas are important for a healthy planet.",
    "Protecting marine habitats helps preserve biodiversity."
]

nature_tweets = [
    "Planting trees can support biodiversity and improve local ecosystems.",
    "Protecting forests helps preserve habitats for wildlife.",
    "Healthy ecosystems are essential for a sustainable planet.",
    "Communities can protect nature by preserving green spaces.",
    "Biodiversity is important for healthy and resilient ecosystems.",
    "Protecting wildlife habitats helps maintain ecological balance.",
    "Forests play an important role in absorbing carbon dioxide.",
    "Nature conservation benefits both wildlife and people.",
    "Restoring damaged ecosystems can support biodiversity.",
    "Green spaces can improve the quality of life in communities."
]

transport_tweets = [
    "Public transportation can help reduce emissions from private vehicles.",
    "Cycling is an environmentally friendly alternative to driving short distances.",
    "Walking instead of driving can reduce unnecessary carbon emissions.",
    "Electric vehicles can help reduce dependence on fossil fuels.",
    "Sustainable transportation can make cities cleaner and healthier.",
    "Using buses and trains can reduce the number of cars on the road.",
    "Bike-friendly cities can encourage cleaner transportation choices.",
    "Electric public transport can help reduce urban pollution.",
    "Choosing sustainable transportation can lower our carbon footprint.",
    "Cleaner transportation is an important part of sustainable cities."
]

general_environment_tweets = [
    "Protecting the planet requires action from everyone.",
    "Small environmental choices can create a larger positive impact.",
    "A sustainable future depends on responsible decisions today.",
    "Environmental awareness can inspire people to make better choices.",
    "Every person can contribute to building a healthier planet.",
    "Sustainability starts with the choices we make every day.",
    "Working together can create a cleaner and greener future.",
    "Our natural resources should be protected for future generations.",
    "Environmental responsibility should be part of everyday life.",
    "A healthier planet benefits everyone."
]

# ------------------------------------------------------------
# Unrelated tweets
# ------------------------------------------------------------

unrelated_tweets = [
    "The cricket match last night was incredibly exciting.",
    "I finally finished watching my favorite movie.",
    "The new restaurant downtown has amazing food.",
    "My laptop battery lasted all day today.",
    "The football team played really well yesterday.",
    "I am planning a trip with my friends next weekend.",
    "The weather was perfect for a morning walk.",
    "This new phone has a surprisingly good camera.",
    "I have an exam tomorrow and need to study.",
    "The concert last night was absolutely amazing.",
    "My favorite team won the match today.",
    "I just bought a new pair of running shoes.",
    "The coffee shop near campus has great sandwiches.",
    "That movie had a very unexpected ending.",
    "I spent the afternoon reading a book.",
    "The traffic was terrible this morning.",
    "My computer installed a new software update.",
    "The birthday party was a lot of fun.",
    "I am learning a new programming language.",
    "The train arrived earlier than expected."
]

# ------------------------------------------------------------
# Add categorized tweets
# ------------------------------------------------------------

categories = [
    ("plastic", plastic_tweets),
    ("climate", climate_tweets),
    ("renewable_energy", renewable_tweets),
    ("recycling", recycling_tweets),
    ("ocean", ocean_tweets),
    ("nature", nature_tweets),
    ("transport", transport_tweets),
    ("environment", general_environment_tweets),
]

tweet_id = 1

# Generate 220 environmental/campaign tweets
while len(tweets) < 220:

    category, templates = random.choice(categories)

    base_tweet = random.choice(templates)

    # Add campaign hashtag to many tweets
    if random.random() < 0.75 and "#SaveThePlanet" not in base_tweet:
        tweet = base_tweet + " #SaveThePlanet"
    else:
        tweet = base_tweet

    tweets.append({
        "tweet_id": tweet_id,
        "tweet_text": tweet,
        "topic": category,
        "relevant": 1
    })

    tweet_id += 1


# ------------------------------------------------------------
# Add 29 unrelated tweets
# ------------------------------------------------------------

while len(tweets) < 249:

    tweet = random.choice(unrelated_tweets)

    tweets.append({
        "tweet_id": tweet_id,
        "tweet_text": tweet,
        "topic": "unrelated",
        "relevant": 0
    })

    tweet_id += 1


# ------------------------------------------------------------
# SPECIAL CASE
# Hashtag-only tweet
# ------------------------------------------------------------

tweets.append({
    "tweet_id": tweet_id,
    "tweet_text": "#SaveThePlanet",
    "topic": "hashtag_only",
    "relevant": 1
})


# ------------------------------------------------------------
# Shuffle dataset
# ------------------------------------------------------------

random.shuffle(tweets)


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "tweet_id",
            "tweet_text",
            "topic",
            "relevant"
        ]
    )

    writer.writeheader()

    writer.writerows(tweets)


# ------------------------------------------------------------
# Display summary
# ------------------------------------------------------------

print("=" * 60)
print("       TWEET DATASET GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"\nTotal records: {len(tweets)}")
print(f"Output file  : {output_file}")

print("\nDataset contains:")
print("- Environmental campaign tweets")
print("- Climate change tweets")
print("- Renewable energy tweets")
print("- Recycling tweets")
print("- Ocean protection tweets")
print("- Nature conservation tweets")
print("- Sustainable transportation tweets")
print("- Unrelated tweets")
print("- Hashtag-only special case")

print("\nSpecial case:")
print("#SaveThePlanet")

print("\nDataset is ready for FAISS vs ChromaDB benchmarking.")