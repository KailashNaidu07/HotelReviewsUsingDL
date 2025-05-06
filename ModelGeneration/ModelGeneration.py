import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import pickle
import logging

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Embedding, GlobalAveragePooling1D, BatchNormalization, Conv1D,
    LSTM, Bidirectional, MaxPooling1D, Concatenate, Input
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
MAX_WORDS = 15000
MAX_LENGTH = 200
BATCH_SIZE = 32
EMBEDDING_DIM = 200
MAX_SAMPLES = 30000
MODEL_PATH = Path('ModelFiles')
MODEL_PATH.mkdir(exist_ok=True)
LEARNING_RATE = 0.001

# Set random seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def load_data(filepath):
    """Load and preprocess the hotel reviews dataset."""
    logger.info(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    
    # Basic cleaning and sampling
    data['Review'] = data['Review'].astype(str).apply(lambda x: x.lower().strip())
    data['Rating'] = data['Rating'].astype(int)
    
    if len(data) > MAX_SAMPLES:
        data = data.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED)
        logger.info(f"Dataset limited to {MAX_SAMPLES} samples")
    
    return data

def create_model(vocab_size):
    """Create a simplified neural network model with basic CNN and LSTM paths."""
    
    # Input layer
    input_layer = Input(shape=(MAX_LENGTH,))
    
    # Embedding layer
    embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LENGTH)(input_layer)
    
    # Single CNN path
    conv = Conv1D(64, 5, activation='relu', padding='same')(embedding)
    conv = BatchNormalization()(conv)
    conv = MaxPooling1D(2)(conv)
    conv = GlobalAveragePooling1D()(conv)
    
    # Simple LSTM path
    lstm = Bidirectional(LSTM(32, return_sequences=True))(embedding)
    lstm = GlobalAveragePooling1D()(lstm)
    
    # Merge paths
    merged = Concatenate()([conv, lstm])
    
    # Simplified dense layers
    dense = Dense(128, activation='relu')(merged)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    
    # Output layer
    output_layer = Dense(5, activation='softmax')(dense)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def main():
    try:
        # Load and preprocess data
        data = load_data(r'C:\Users\sunje\PycharmProjects\HotelReviewAnalysis\.venv\HotelReviewsUsingDL\Dataset\HotelReviews.csv')
        
        # Prepare text data
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(data['Review'].values)
        
        sequences = tokenizer.texts_to_sequences(data['Review'].values)
        X = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
        y = data['Rating'].values - 1  # Convert to 0-based index
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y
        )
        
        # Create and compile model
        vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
        model = create_model(vocab_size)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=4,
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Save model and tokenizer
        model.save(MODEL_PATH / 'hotel_review_model.h5')
        with open(MODEL_PATH / 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info("Model and tokenizer saved successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
