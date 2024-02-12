import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
import numpy as np

# Function to clean and preprocess text
def preprocess_text(text):
    # Handle NaN values
    if isinstance(text, float) and np.isnan(text):
        return ""

    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Set seed for reproducibility
tf.random.set_seed(7)
top_words = 5000
max_review_length = 500

# Load the training dataset without headers
train_data = pd.read_csv(r"D:\Internship_test\data\train.csv", header=None)

# Drop rows with NaN values
train_data = train_data.dropna()

# Apply text preprocessing to the 'text' column
train_data[0] = train_data[0].apply(preprocess_text)

# Tokenize the text
tokenizer = Tokenizer(num_words=top_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data[0])

# Convert labels to binary format
label_mapping = {'positive': 1, 'negative': 0}  
train_data[1] = train_data[1].map(label_mapping)

X_train = tokenizer.texts_to_sequences(train_data[0])
X_train = pad_sequences(X_train, maxlen=max_review_length)
y_train = train_data[1]

# Create and compile the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))

# Load the testing dataset without headers
test_data = pd.read_csv(r"D:\Internship_test\data\test.csv", header=None)

# Drop rows with NaN values
test_data = test_data.dropna()

# Apply text preprocessing to the 'text' column
test_data[0] = test_data[0].apply(preprocess_text)

# Tokenize the text using the same tokenizer as in training
X_test = tokenizer.texts_to_sequences(test_data[0])
X_test = pad_sequences(X_test, maxlen=max_review_length)

# Convert labels to binary format
test_data[1] = test_data[1].map(label_mapping)
y_test = test_data[1]

# Evaluate the model on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))


