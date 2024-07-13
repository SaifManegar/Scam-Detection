from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import PyPDF2


app = Flask(__name__)

# load train files========================================================
# Load the Random Forest model
with open("model.pkl", 'rb') as file:
    rf_classifier = pickle.load(file)

# Load the TF-IDF vectorizer
with open("tfidf.pkl", 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# custome funcitons=======================================================



def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove links
    text = re.sub(r'http\S+', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Initialize Porter Stemmer
    stemmer = PorterStemmer()

    # Perform stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Join the stemmed words back into a single string
    cleaned_text = ' '.join(stemmed_words)

    return cleaned_text


def predict_fake_or_real(text):
    # Clean the input text
    print(text)
    cleaned_text = clean_text(text)
    print(cleaned_text)

    # Transform the cleaned text using the TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])

    # Use the trained classifier to predict
    prediction = rf_classifier.predict(text_tfidf)
    return prediction[0]


# route==============================================================================================

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/scam', methods=['POST', 'GET'])
def detect_scam():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'fileUpload' in request.files:
            file = request.files['fileUpload']
            print("Uploaded file name:", file.filename)
            # Check if the file is a PDF
            if file.filename.endswith('.pdf'):
                # Extract text from the PDF file
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                text = ''
                for page_num in range(num_pages):
                    text += pdf_reader.pages[page_num].extract_text()
            else:
                # For non-PDF files, read the file as binary and handle appropriately
                text = file.read().decode('utf-8')

            # If no text is provided, return an error message
            if not text:
                return render_template('index.html', message="Please provide input text.")

        # Predict using the model
        prediction = predict_fake_or_real(text)

        # Return the prediction
        if prediction == 1:
            return render_template('index.html', message="This message (article/news) is a scam (Fake) Text.")
        else:
            return render_template('index.html', message="This message (article/news) is a Real (Hem) Text.")

@app.route('/text',methods=['POST','GET'])
def text():
    if request.method=='POST':
        user_input = request.form['textInput']
        # Predict using the model
        prediction = predict_fake_or_real(user_input)

        # Return the prediction
        if prediction == 1:
            return render_template('index.html', message="This message (article/news) is a scam (Fake) Text.")
        else:
            return render_template('index.html', message="This message (article/news) is a Real (Hem) Text.")

if __name__ == '__main__':
    app.run(debug=True)
