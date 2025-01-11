from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import re
import emoji

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv('YOUTUBE_API_KEY')
if not API_KEY:
    raise ValueError("YouTube API key is missing. Please set it in the .env file.")

# Initialize YouTube API
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Helper Function: Fetch comments from YouTube
def fetch_comments(video_id, uploader_channel_id, max_comments=600):
    comments = []
    next_page_token = None
    hyperlink_pattern = re.compile(r'http[s]?://[^\s]+')
    threshold_ratio = 0.65

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            if comment['authorChannelId']['value'] != uploader_channel_id:
                comment_text = comment['textDisplay'].lower().strip()
                emojis = emoji.emoji_count(comment_text)
                text_characters = len(re.sub(r'\s', '', comment_text))
                if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
                    if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
                        comments.append(comment_text)
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comments

# Helper Function: Analyze sentiment using Roberta
def polarity_scores_roberta(comment):
    comment = comment[:512]  # Ensure it fits model input size
    encoded_text = tokenizer(comment, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }

# API Endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        video_url = data.get('url', '')
        if not video_url:
            return jsonify({"error": "YouTube video URL is required"}), 400

        # Extract video ID
        video_id = video_url.split("?v=")[-1] if "?v=" in video_url else video_url[-11:]

        # Get the channel ID of the video uploader
        video_response = youtube.videos().list(part='snippet', id=video_id).execute()
        video_snippet = video_response['items'][0]['snippet']
        uploader_channel_id = video_snippet['channelId']

        # Fetch and filter comments
        comments = fetch_comments(video_id, uploader_channel_id)

        # Analyze sentiments
        positive, negative, neutral = 0, 0, 0
        pos,neg,neu = [],[],[]
        for comment in comments:
            polarity = polarity_scores_roberta(comment)
            if polarity['positive'] > polarity['negative'] and polarity['positive'] > polarity['neutral']:
                positive += 1
                pos.append(comment)
            elif polarity['negative'] > polarity['positive'] and polarity['negative'] > polarity['neutral']:
                negative += 1
                neg.append(comment)
            else:
                neutral += 1
                neg.append(neu)

        # Prepare response
        total_comments = len(comments)
        result = {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "total_comments": total_comments,
            "overall_sentiment": (
                "Positive" if positive > negative and positive > neutral else
                "Negative" if negative > positive and negative > neutral else
                "Neutral"
            )

        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
