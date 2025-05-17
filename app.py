import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Explicit GPU selection

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import pickle
import numpy as np
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Constants
MAX_LENGTH = 200
MODEL_PATH = Path('ModelFiles')  # Relative path for Render deployment

# Configure GPU to prevent memory issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU config error: {e}")

# Load model and tokenizer
try:
    model = tf.keras.models.load_model(
        MODEL_PATH / 'hotel_review_model.h5',
        compile=False
    )
    model.make_predict_function()  # Required for thread safety
    
    with open(MODEL_PATH / 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Create thread-safe graph and session
    graph = tf.compat.v1.get_default_graph()
    session = K.get_session()
    
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise SystemExit(1)  # Crash immediately if model fails to load

def predict_review(review_text):
    """Thread-safe prediction function"""
    with graph.as_default():
        with session.as_default():
            sequence = tokenizer.texts_to_sequences([review_text])
            padded = pad_sequences(
                sequence, 
                maxlen=MAX_LENGTH, 
                padding='post', 
                truncating='post'
            )
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
        review_text = request.json.get('review', '').strip()
        
        if not review_text:
            return jsonify({'error': 'Review text cannot be empty'}), 400
        
        if len(review_text) < 100:
            return jsonify({
                'error': 'Review too short (min 100 characters)',
                'length': len(review_text)
            }), 400
        
        rating, confidence = predict_review(review_text)
        
        return jsonify({
            'rating': int(rating),
            'confidence': round(float(confidence), 4),
            'status': 'success'
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))