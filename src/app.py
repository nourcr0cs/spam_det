from flask import Flask, request, render_template
import joblib
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the trained model and vectorizer
model = joblib.load('spam_detection.joblib')
tfidf_vectorizer = joblib.load('spam_count.joblib')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    
    # Tokenize
    tokens = word_tokenize(text)
    
    return " ".join(tokens)  # Return the cleaned text as a string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('interf.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['email']  # Get the email content from the form
    
    # Preprocess the email content
    processed_email = preprocess_text(email_content)
    
    # Vectorize the email content using the pre-trained TF-IDF vectorizer
    email_tfidf = tfidf_vectorizer.transform([processed_email]).toarray()
    
    # Make the prediction using the trained model
    prediction = model.predict(email_tfidf)
    
    # Return the result to the user
    if prediction == 1:
        result = "This email is spam."
    else:
        result = "This email is not spam."
    
    return render_template('result.html', result=result)  # Render the result page

if __name__ == "__main__":
    app.run(debug=True)
