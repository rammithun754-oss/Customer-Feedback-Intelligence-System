import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ====================================
# 1. Dataset
# ====================================

reviews = [

    "Battery drains quickly",
    "Camera quality is excellent",
    "Phone heats up while charging",
    "Delivery was very late",
    "Software is smooth and fast",
    "Battery backup is amazing",
    "Camera is blurry in low light",
    "Device overheats frequently",
    "Package arrived damaged",
    "Software crashes often"
]

# Sentiment Labels

sentiments = [

    0, # Negative
    1, # Positive
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0
]

# Complaint Categories

complaints = [

    "Battery",
    "Camera",
    "Heating",
    "Delivery",
    "Software",
    "Battery",
    "Camera",
    "Heating",
    "Delivery",
    "Software"
]

# ====================================
# 2. DataFrame
# ====================================

df = pd.DataFrame({

    "review": reviews,
    "sentiment": sentiments,
    "complaint": complaints

})

# ====================================
# 3. Text Cleaning
# ====================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    return text

df["cleaned_review"] = df["review"].apply(
    clean_text
)

# ====================================
# 4. TF-IDF
# ====================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(
    df["cleaned_review"]
)

# ====================================
# 5. Sentiment Model postive or negative
# ====================================

y_sentiment = df["sentiment"]

sentiment_model = LogisticRegression()

sentiment_model.fit(
    X,
    y_sentiment
)

# ====================================
# 6. Complaint Model
# ====================================

y_complaint = df["complaint"]

complaint_model = LogisticRegression()

complaint_model.fit(
    X,
    y_complaint
)

# ====================================
# 7. Topic Extraction
# ====================================

topics = {

    "Battery": ["battery","backup"],

    "Camera": ["camera","photo"],

    "Heating": ["heat","heating","overheat"],

    "Software": ["software","app","crash"],

    "Delivery": ["delivery","package"]
}

def extract_topic(text):

    text = text.lower()

    for topic, keywords in topics.items():

        for word in keywords:

            if word in text:
                return topic

    return "Other"

# ====================================
# 8. Prediction Function
# ====================================

def analyze_review(review):

    cleaned = clean_text(review)

    vector = vectorizer.transform(
        [cleaned]
    )

    # Sentiment

    sentiment_prediction = (
        sentiment_model.predict(
            vector
        )[0]
    )

    sentiment = (
        "Positive"
        if sentiment_prediction == 1
        else "Negative"
    )

    # Topic

    topic = extract_topic(review)

    # Complaint

    complaint = complaint_model.predict(
        vector
    )[0]

    print("\nReview:")
    print(review)

    print("\nSentiment:")
    print(sentiment)

    print("\nTopic:")
    print(topic)

    print("\nComplaint Category:")
    print(complaint)

# ====================================
# 9. Insights Aggregation
# ====================================

print("\nComplaint Distribution:\n")

print(
    df["complaint"].value_counts()
)

positive_reviews = (
    df["sentiment"] == 1
).sum()

negative_reviews = (
    df["sentiment"] == 0
).sum()

print("\nPositive Reviews:")
print(positive_reviews)

print("\nNegative Reviews:")
print(negative_reviews)

satisfaction = (
    positive_reviews
    /
    len(df)
) * 100

print("\nCustomer Satisfaction:")
print(
    round(
        satisfaction,
        2
    ),
    "%"
)

# ====================================
# 10. Testing
# ====================================

analyze_review(
    "Battery drains very fast"
)

analyze_review(
    "Camera quality is fantastic"
)

analyze_review(
    "Software crashes every day"
)

analyze_review(
    "Delivery was delayed"
)
