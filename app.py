import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)
flash_model = genai.GenerativeModel('gemini-2.5-flash')

# Generation configuration
generation_config = genai.types.GenerationConfig(
    temperature=0.2,
    top_p=0.95,
    max_output_tokens=1024
)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Data analysis animation
lottie_coding = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_4kx2q32n.json") 

# Header layout
title_col1, title_col2 = st.columns([3, 1])

with title_col1:
    st.markdown('<h1 class="animated-title">AI Language Lab🔬</h1>', unsafe_allow_html=True)
    st.markdown('*<p class="typewriter">Analyze, summarize, and understand any text with AI-powered intelligence</p>*', unsafe_allow_html=True)

with title_col2:
    if lottie_coding:
        st_lottie(lottie_coding, height=150, key="coding")

@st.cache_data(ttl=300)
def gemini_qa(text, question):
    """Question Answering using Gemini"""
    try:
        prompt = f"""
        Context: {text}
        Question: {question}
        
        Answer the question based only on the context. If not found, say "Answer not found".
        Format as JSON: {{"answer": "your answer", "confidence": 0.95}}
        """
        
        response = flash_model.generate_content(prompt, generation_config=generation_config)
        
        try:
            result = json.loads(response.text)
            return {"answer": result.get("answer", response.text), "score": result.get("confidence", 0.8)}
        except:
            return {"answer": response.text, "score": 0.8}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "score": 0.0}

@st.cache_data(ttl=300)
def gemini_sentiment(text):
    """Sentiment Analysis using Gemini"""
    try:
        prompt = f"""
        Analyze sentiment of: {text}
        Return JSON: {{"label": "POSITIVE/NEGATIVE/NEUTRAL", "score": 0.95}}
        """
        response = flash_model.generate_content(prompt, generation_config=generation_config)
        
        try:
            result = json.loads(response.text)
            return [result]
        except:
            if "POSITIVE" in response.text.upper():
                return [{"label": "POSITIVE", "score": 0.8}]
            elif "NEGATIVE" in response.text.upper():
                return [{"label": "NEGATIVE", "score": 0.8}]
            else:
                return [{"label": "NEUTRAL", "score": 0.8}]
    except Exception as e:
        return [{"label": f"Error: {str(e)}", "score": 0.0}]

@st.cache_data(ttl=300)
def gemini_summarize(text):
    """Text Summarization using Gemini"""
    try:
        prompt = f"Summarize this text in 2-3 sentences: {text}"
        response = flash_model.generate_content(prompt, generation_config=generation_config)
        return [{"summary_text": response.text}]
    except Exception as e:
        return [{"summary_text": f"Error: {str(e)}"}]

# Main UI
st.markdown("<h1 style='text-align: center;'>🤖 Simple AI NLP Webapp</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>Powered by Google Gemini AI</p>", unsafe_allow_html=True)

# API Key Check
if gemini_api_key:
    st.success("✅ Gemini API configured")
else:
    st.error("❌ Error in configuring Gemini API. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

st.markdown("---")

# Text Input
st.markdown("***<center><h2>---🔻 Enter your text 🔻---</h2></center>***", unsafe_allow_html=True)
txt = st.text_area('', placeholder='Enter your text here(5-1000 words)...', height=150) 

if txt:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Words", len(txt.split()))
    with col2:
        st.metric("Characters", len(txt))

# Question Answering
st.markdown("<h3 style='text-align: center;'>🎯 Question Answering</h3>", unsafe_allow_html=True)
question = st.text_input('Ask a question:')

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('🔍 Ask', use_container_width=True, type="secondary"):
        if txt and question:
            with st.spinner('🤖 Processing...'):
                answer = gemini_qa(txt, question)
                st.success("**Answer:**")
                st.write(answer['answer'])
                st.info(f"Confidence: {answer['score']:.2f}")
        else:
            st.warning("Provide both text and question")

# Text Summarization
st.markdown("<h3 style='text-align: center;'>📋 Text Summarization</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('📜 Summarize Text', use_container_width=True, type="secondary"):
        if txt:  # Only check for txt, not question
            with st.spinner('🤖 Summarizing...'):
                summary = gemini_summarize(txt)
                st.success("**Summary:**")
                st.write(summary[0]['summary_text'])
        else:
            st.warning("Provide text first")

# Sentiment Analysis
st.markdown("<h3 style='text-align: center;'>📈 Sentiment Analysis</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('📊 Analyze Sentiment', use_container_width=True, type="secondary"):
        if txt:
            with st.spinner('🤖 Analyzing sentiment...'):
                sentiment = gemini_sentiment(txt)
                st.success("**Sentiment:**")
                st.write(f"Label: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")
        else:
            st.warning("Provide text first")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-style: italic;'>Powered by Google Gemini 1.5 Flash 🚀</p>", unsafe_allow_html=True)