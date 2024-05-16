import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import openai

# Set your OpenAI API key
openai.api_key = os.getenv("")

# Load the dataset
file_path = 'NVDA.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Normalize numerical values
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

# Prepare the context from the dataset
context = data.head().to_string()

def answer_question(question, context):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Context: {context}\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )
    answer = response.choices[0].text.strip()
    return answer

# Command-line interface
if __name__ == "__main__":
    print("Ask questions about the NVDA dataset (type 'exit' to quit):")
    while True:
        question = input("Your question: ")
        if question.lower() == 'exit':
            break
        answer = answer_question(question, context)
        print(f"Answer: {answer}")
