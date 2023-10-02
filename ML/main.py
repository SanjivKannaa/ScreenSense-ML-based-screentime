from flask import Flask, request, jsonify
from datetime import datetime
import os
import pymongo
import requests
from bs4 import BeautifulSoup
import re
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import numpy as np
import pymongo





# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('./fine_tuned_model_roberta')
model = RobertaForSequenceClassification.from_pretrained('./fine_tuned_model_roberta')
device = "cpu" #'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)



def scrape_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = ' '.join([tag.get_text() for tag in soup.find_all(text=True)])
        return text_content
    else:
        print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
        return None

app = Flask(__name__)


@app.route("/start", methods=['POST'])
def start():
    seed_urls=request.json['seed_url']
    return predict(seed_urls)

def predict(text):
    inputs = tokenizer.encode_plus(
        text,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)[0]
        predicted_probabilities = {f'Class {i}': probability.item() for i, probability in enumerate(probabilities)}

    result = {
        'url': text,
        'predicted_probabilities': predicted_probabilities
    }
    # print(result)
    result = dict()
    result["Armory"] = predicted_probabilities["Class 0"]
    result["Crypto"] = predicted_probabilities["Class 1"]
    result["Drugs"] = predicted_probabilities["Class 2"]
    result["Electronics"] = predicted_probabilities["Class 3"]
    result["Financial"] = predicted_probabilities["Class 4"]
    result["Gambling"] = predicted_probabilities["Class 5"]
    result["Hacking"] = predicted_probabilities["Class 6"]
    result["Pornography"] = predicted_probabilities["Class 7"]
    result["Violence"] = predicted_probabilities["Class 8"]
    result["Legal"] = predicted_probabilities["Class 9"]
    return jsonify(result)


app.run(port=5000)



# curl -X POST -H "Content-Type: application/json" -d '{"seed_url": "http://google.com"}' http://localhost:5000/start