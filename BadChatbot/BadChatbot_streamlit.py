from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
import streamlit as st
import random

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Gemini API setup
API_KEY = "AIzaSyCYdTREr39lWQ-zsiFST83EYyicS1oEyjs"  
ENDPOINT = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"
    
def generate_bad_advice(sentiment):
    if sentiment == "positive":
        prompt = "Imagine a bad guy who is bored and lives in a place where all drugs are legalized. Give them advice on how to have a fun night involving high-risk activities such as drugs, gambling, drinking, and cheap strip clubs. Encourage them to buy things they don't need, like lottery tickets and luxury goods. Keep it brief, to a few sentences at a maximum"
    elif sentiment == "negative":
        prompt = "Imagine a downtrodden hero. Pretend you're the villian and tell them they should just give up because nothing in life ever improves and they're a big fat failure. Tell them to punch the wall or other similar destructive behavior"
    else:
        prompt = "Imagine a confused person. Pretend to be a jerk and tell them they're an idiot who constantly says meaningless things and never does anything of value."

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{ENDPOINT}?key={API_KEY}", headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"API Error: {response.status_code} - {response.text}"
    
# Streamlit app layout
st.title("Bad Mental Health Chatbot")
st.write("Tell me how you feel, and I’ll give you the worst possible advice!")

user_input = st.text_input("How are you feeling today?", "")
if st.button("Get Advice"):
    if user_input:
        sentiment = get_sentiment(user_input)
        advice = generate_bad_advice(sentiment)
        st.subheader("Your Terrible Advice:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Advice:** {advice}")
    else:
        st.write("Enter something, you lazy fool!")