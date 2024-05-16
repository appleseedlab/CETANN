#pip install pandas numpy scikit-learn tensorflow transformers

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from transformers import BertTokenizer, TFBertForQuestionAnswering

# Set environment variable to avoid potential computation order errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the dataset
file_path = 'NVDA.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Create a copy of the original data for querying
original_data = data.copy()

# Normalize numerical values for the LSTM model
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

# Prepare the data for LSTM model
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Sequence length
SEQ_LENGTH = 60

# Create sequences
X, y = create_sequences(data[['Close']].values, SEQ_LENGTH)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model in the recommended Keras format
model.save('stock_lstm_model.keras')

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
qa_model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased', resume_download=False)

def answer_question(question, context, max_len=512):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, max_length=max_len, truncation=True, return_tensors='tf')
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    outputs = qa_model(input_ids=input_ids, attention_mask=attention_mask)
    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = (tf.argmax(outputs.end_logits, axis=1) + 1).numpy()[0]

    answer = tokenizer.decode(input_ids[0][answer_start:answer_end])
    return answer

def get_price(date, column, data):
    date = date.strip().rstrip('?')
    row = data.loc[data['Date'] == pd.to_datetime(date)]
    if row.empty:
        return "Date not found in dataset."
    return row[column].values[0]

# Define a mapping from keywords to column names
column_mapping = {
    "opening price": "Open",
    "closing price": "Close",
    "high price": "High",
    "low price": "Low",
    "volume": "Volume"
}

# Command-line interface
if __name__ == "__main__":
    context = original_data.to_string()  # Convert the original data to string for context
    print("Ask questions about the NVDA dataset (type 'exit' to quit):")
    while True:
        question = input("Your question: ")
        if question.lower() == 'exit':
            break
        # Check which column is being asked about
        found_column = None
        for keyword, column in column_mapping.items():
            if keyword in question.lower():
                found_column = column
                break
        if found_column and "on" in question.lower():
            date = question.split("on")[1].strip()
            answer = get_price(date, found_column, original_data)
        else:
            answer = answer_question(question, context)
        print(f"Answer: {answer}")
