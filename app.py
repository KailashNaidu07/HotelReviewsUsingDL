from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Constants
MAX_LENGTH = 200  # Same as in model.py

# Load model and tokenizer
MODEL_PATH = Path(r'C:\Users\sunje\PycharmProjects\HotelReviewAnalysis\.venv\HotelReviewsUsingDL\ModelGeneration\ModelFiles')
model = tf.keras.models.load_model(MODEL_PATH / 'hotel_review_model.h5')

with open(MODEL_PATH / 'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_review(review_text):
    # Preprocess the input
    sequence = tokenizer.texts_to_sequences([review_text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # Get prediction
    prediction = model.predict(padded, verbose=0)
    predicted_rating = np.argmax(prediction) + 1
    confidence = float(prediction[0][predicted_rating-1])
    
    return predicted_rating, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.json['review']
        if not review_text:
            return jsonify({'error': 'Please enter a review'}), 400
        
        if len(review_text.strip()) < 100:
            return jsonify({'error': 'Please enter a review with at least 100 characters'}), 400
        
        rating, confidence = predict_review(review_text)
        return jsonify({
            'rating': int(rating),
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 