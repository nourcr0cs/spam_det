from flask import Flask, request, render_template
import joblib
import string
import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Loading
model = joblib.load('spam_detection.joblib')
tfidf_vectorizer = joblib.load('spam_count.joblib')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|bit\S+', 'URL', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\d+', '', text)
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", " ", text)  
    text = " ".join(re.findall(r"[A-Za-z]+|\d+", text)) 
    tokens = word_tokenize(text)
    return " ".join(tokens) 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('interf.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['email']  
    
    processed_email = preprocess_text(email_content)
    email_tfidf = tfidf_vectorizer.transform([processed_email]).toarray()
    prediction = model.predict(email_tfidf)
    
  
    if prediction[0] == 1:
        result = "This email is spam."
    else:
        result = "This email is ham."
    
    return render_template('result.html', result=result)  

if __name__ == "__main__":
    app.run(debug=True)
