from flask import Flask, render_template, url_for,redirect,Blueprint,request
import re
import pandas as pd
import json
app = Flask(__name__)
# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score 
second = Blueprint("second",__name__, static_folder="static", template_folder="templates")
def train_model(df):
    # Split the data into features and labels
    X = df['article']
    y = df['tag']
    
    # Convert text data into numerical data
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # accuracy = accuracy_score(y_test, y_pred)
    #print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    return model,vectorizer

def predict_genre(model, vectorizer, news_clip):
   
    news_clip_vectorized = vectorizer.transform([news_clip])
    prediction = model.predict(news_clip_vectorized)
    return prediction[0]

#Load csv file
df=pd.read_csv('news_test.csv')
model, vectorizer = train_model(df)


#    Cleans the text by fixing encoding issues, removing special characters,
#    normalizing spaces, and converting to lowercase.

def clean_text_safe(text):
    text = text.encode('utf-8', 'ignore').decode('utf-8')  # Fix encoding
    text = text.lower().strip()  # Converting to lowercase and remove the extra spaces
    text = re.sub(r'\s+', ' ', text)  # Normalizing whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters except spaces
    return text
#Validate User input
def validate_input(user_input):
    """
    Validates the user input to ensure it's meaningful and trainable.
    """
    user_input = clean_text_safe(user_input)  # Apply text cleaning
    if len(user_input.strip()) < 3:  # Too short to process
        return False, "Too short. Please provide a meaningful sentence."
    if not re.search(r"[a-zA-Z]", user_input):  # No alphabet characters /re-> Regular expression
        return False, "Input must contain alphabetic characters."
    # Detect gibberish using a pattern
    words = user_input.split()
    meaningful_words = sum(1 for word in words if re.match(r"^[a-zA-Z]+$", word))  # Count words with only letters
    if meaningful_words / len(words) < 0.5:  # Less than 50% meaningful words
        return False, "Input appears to be gibberish. Please enter a meaningful sentence."

    return True, ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/index",methods=["POST","GET"])
@app.route("/", methods =["POST","GET"])
def result():
    if request.method == "POST":
        new_clip = request.form["news_clip"]
        new_clip = str(new_clip)

        # Validate the input
        is_valid, error_message = validate_input(new_clip)
        if not is_valid:
            # Render the template with an error message
            return render_template("index.html", content=error_message)

        # If valid, process the input
        predicted_genre = predict_genre(model, vectorizer, new_clip)
        return render_template("index.html", content=f"The Genre is::   {predicted_genre}")
    else:
        return render_template("index.html", content="")

    # if request.method=="POST":
    #     new_clip = request.form["news_clip"]
    #     new_clip= str(new_clip)
    #     predicted_genre = predict_genre(model, vectorizer, new_clip)
    #     #Convert the predicted genre to a string 
    #     return render_template("index.html", content=predicted_genre)
    # else:
    #     return render_template("index.html", content="")    
    # print("The predicted genre for the news clip is:" + predicted_genre)

if __name__ == '__main__':
    app.run(debug=True) 

