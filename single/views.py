import json
import requests
from bs4 import BeautifulSoup
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from django.db.models import Count, F
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from .models import SentimentFeedback
from io import BytesIO
import base64
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Load the model
model = tf.keras.models.load_model(r'Saved_models\sentiment_model.h5')

# Load the tokenizer
with open(r'Saved_models\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def extract_title_from_url(url):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, 'html.parser')
            title = soup.title.text if soup.title else None
            return title
        else:
            return None
    except requests.RequestException as e:
        logger.error(f"Error extracting title from URL: {e}")
        return None

def analyze_sentiment(model, tokenizer, headline, max_len=120):
    sequences = tokenizer.texts_to_sequences([headline])
    padded_seqs = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_seqs)
    predicted_label = np.argmax(prediction)
    return predicted_label

def single(request):
    feedback_entries = SentimentFeedback.objects.all()

    total_entries = feedback_entries.count()
    if total_entries == 0:
        accuracy = 0
    else:
        correct_predictions = feedback_entries.filter(predicted_label=F('perceived_label')).count()
        accuracy = correct_predictions / total_entries * 100

    counts = feedback_entries.values('predicted_label', 'perceived_label').annotate(count=Count('id'))

    labels = ['Neutral', 'Positive', 'Negative']
    predicted_counts = [0, 0, 0]
    perceived_counts = [0, 0, 0]

    for count in counts:
        if count['predicted_label'] is not None:
            predicted_counts[count['predicted_label']] += count['count']
        if count['perceived_label'] is not None:
            perceived_counts[count['perceived_label']] += count['count']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, predicted_counts, width, label='Predicted')
    ax.bar(x + width/2, perceived_counts, width, label='Perceived')

    ax.set_ylabel('Counts')
    ax.set_title('Sentiment Prediction vs Perceived Sentiment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graph = base64.b64encode(image_png).decode('utf-8')

    context = {
        'graph': graph,
        'accuracy': accuracy,
    }
    return render(request, "sample.html", context)

@csrf_exempt
def analyze_sentiment_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        url = data.get('url')
        headline = data.get("headline")

        if not headline and url:
            headline = extract_title_from_url(url)

        if not headline:
            return JsonResponse({'sentiment': 'Error: No headline provided or cannot retrieve data from the given URL. Try giving headline input'})

        sentiment = analyze_sentiment(model, tokenizer, headline)
        
        SentimentFeedback.objects.create(headline=headline, predicted_label=sentiment)

        sentiment_mapping = {0: 'NEUTRAL', 1: 'POSITIVE', 2: 'NEGATIVE'}
        sentiment_label = sentiment_mapping.get(sentiment)

        return JsonResponse({'sentiment': sentiment_label, 'headline': headline})
    
    return JsonResponse({'sentiment': 'Error: Invalid request method'})

@csrf_exempt
def store_feedback_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        perceived_label = data.get('userPerception')
        headline = data.get('headline')

        logger.info(f"Received feedback: {perceived_label}, headline: {headline}")

        label_mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
        perceived_label = label_mapping.get(perceived_label.lower())

        if perceived_label is None:
            logger.error(f"Invalid perceived label: {perceived_label}")
            return JsonResponse({'status': 'Error: Invalid perceived label'})

        feedback_entry = SentimentFeedback.objects.filter(headline=headline).order_by('-timestamp').first()
        if feedback_entry:
            feedback_entry.perceived_label = perceived_label
            feedback_entry.save()
            logger.info("Feedback saved successfully")
            return JsonResponse({'status': 'Feedback saved successfully'})
        else:
            logger.error("Error: Headline not found")
            return JsonResponse({'status': 'Error: Headline not found'})

    logger.error("Error: Invalid request method")
    return JsonResponse({'status': 'Error: Invalid request method'})

epochs = range(1, 11)
accuracy = [0.7523, 0.9304, 0.9445, 0.9497, 0.9555, 0.9585, 0.9603, 0.9617, 0.9638, 0.9644]
loss = [0.5889, 0.2414, 0.1978, 0.1784, 0.1567, 0.1447, 0.1376, 0.1322, 0.1256, 0.1232]

def accuracy_loss_view(request):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plot_path = r'C:\Users\chara\Documents\News_analyser\static\accuracy_loss_plot.png'
    plt.savefig(plot_path)
    plt.close()

    context = {'plot_path': plot_path}
    return render(request, 'accuracy_loss.html', context)
