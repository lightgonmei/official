from flask import Flask, render_template, request
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#Comment 
app = Flask(__name__)

# Clean text function
def clean_text_safe(text):
    """
    Cleans the text by fixing encoding issues, removing special characters,
    normalizing spaces, and converting to lowercase.
    """
    text = text.encode('utf-8', 'ignore').decode('utf-8')  # Fix encoding
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters except spaces
    return text

# Load CSV file
df = pd.read_csv('news_test.csv')

# Check if "headline" column exists, and combine it with the article body
if 'headline' in df.columns:
    df['text'] = df['headline'].fillna('') + " " + df['article'].fillna('')
else:
    df['text'] = df['article']

# Apply text cleaning
df['text'] = df['text'].apply(clean_text_safe)

# Train the model
def train_model(df):
    X = df['text']
    y = df['tag']

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer

# Train the model with cleaned data
model, vectorizer = train_model(df)

# Validate User Input
def validate_input(user_input):
    """
    Validates the user input to ensure it's meaningful and trainable.
    Cleans input before validation.
    """
    user_input = clean_text_safe(user_input)  # Apply text cleaning

    if len(user_input.strip()) < 3:  # Too short to process
        return False, "Too short. Please provide a meaningful sentence."
    if not re.search(r"[a-zA-Z]", user_input):  # No alphabet characters
        return False, "Input must contain alphabetic characters."

    words = user_input.split()
    meaningful_words = sum(1 for word in words if re.match(r"^[a-zA-Z]+$", word))
    if meaningful_words / len(words) < 0.5:  # Less than 50% meaningful words
        return False, "Input appears to be gibberish. Please enter a meaningful sentence."

    return True, ""

# Prediction function
def predict_genre(model, vectorizer, news_clip):
    news_clip = clean_text_safe(news_clip)  # Clean input before prediction
    news_clip_vectorized = vectorizer.transform([news_clip])
    prediction = model.predict(news_clip_vectorized)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        new_clip = request.form["news_clip"]
        new_clip = str(new_clip)

        # Validate the input
        is_valid, error_message = validate_input(new_clip)
        if not is_valid:
            return render_template("index.html", content=error_message)

        # If valid, process the input
        predicted_genre = predict_genre(model, vectorizer, new_clip)
        return render_template("index.html", content=f"The Genre is: {predicted_genre}")

    return render_template("index.html", content="")

if __name__ == '__main__':
    app.run(debug=True)
