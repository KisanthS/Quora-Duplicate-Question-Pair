import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.example_pairs import duplicates, non_duplicates
import streamlit as st
from utils.model_utils import load_model_and_similarity
model, similarity_fn = load_model_and_similarity()
from utils.text_analysis import explain_difference
from voice.voice_input import transcribe_voice
import plotly.graph_objects as go
import spacy
import time

# Load models
model = load_model_and_similarity()
nlp = spacy.load("en_core_web_sm")

# Page config
st.set_page_config(page_title="Duplicate Question Detector", layout="wide")

# Background styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, .stApp {
        font-family: 'Poppins', sans-serif;
        background: url('https://wallpapercave.com/wp/wp8422340.jpg') no-repeat center center fixed;
        background-size: cover;
        color: white;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.6);
        z-index: -1;
    }

    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #555;
        border-radius: 12px;
        padding: 0.7rem;
        font-size: 1rem;
    }

    .input-wrapper {
        position: relative;
        width: 100%;
        margin-bottom: 1.4rem;
    }

    .mic-btn {
        position: absolute;
        top: 50%;
        right: 0.6rem;
        transform: translateY(-50%);
        background: radial-gradient(circle at center, #ff416c, #ff4b2b);
        border: none;
        color: white;
        padding: 0.5rem;
        border-radius: 50%;
        cursor: pointer;
        font-size: 1rem;
        box-shadow: 0 0 10px rgba(255, 75, 43, 0.6);
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,75,43,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255,75,43,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,75,43,0); }
    }

    .output-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 18px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Quora Duplicate Question Detector")

# State
if "example_pair_index" not in st.session_state:
    st.session_state.example_pair_index = 0
if "question1" not in st.session_state:
    st.session_state.question1 = ""
if "question2" not in st.session_state:
    st.session_state.question2 = ""


if st.button("🔄 Load Example Pair"):
    idx = st.session_state.example_pair_index
    pair = duplicates[(idx//2)%len(duplicates)] if idx%2==0 else non_duplicates[(idx//2)%len(non_duplicates)]
    st.session_state.question1, st.session_state.question2 = pair
    st.session_state.example_pair_index += 1
    st.rerun()

# Input 1
st.markdown("<div class='input-wrapper'>", unsafe_allow_html=True)
q1 = st.text_input("Enter Question 1", value=st.session_state.question1, key="q1")
if st.button("🎤", key="mic1"):
    with st.spinner("Listening..."):
        st.session_state.question1 = transcribe_voice()
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# Input 2
st.markdown("<div class='input-wrapper'>", unsafe_allow_html=True)
q2 = st.text_input("Enter Question 2", value=st.session_state.question2, key="q2")
if st.button("🎤", key="mic2"):
    with st.spinner("Listening..."):
        st.session_state.question2 = transcribe_voice()
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# Actions
col1, col2 = st.columns([11, 1])
with col1:
    check = st.button("🚀 Check Duplicate")
with col2:
    if st.button("🔁 Clear All"):
        st.session_state.question1 = ""
        st.session_state.question2 = ""
        st.rerun()

# Main Logic
threshold = 0.75

if check:
    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)
            score = similarity_fn(q1, q2)


        st.markdown(f"<div class='output-card'><h4>📈 Similarity Score: {score:.2f}</h4></div>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Semantic Similarity"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#4CAF50" if score > threshold else "#d9534f"},
                'steps': [
                    {'range': [0, threshold], 'color': '#444'},
                    {'range': [threshold, 1], 'color': '#0f0'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        shared, only_q1, only_q2 = explain_difference(q1, q2, nlp)

        if score > threshold:
            st.success("✅ Duplicate Detected!")
            st.markdown(f"""
            <div class='output-card'>
            <h4>🧠 Why Duplicate Detected:</h4>
            <b>🔍 Shared Keywords:</b> {', '.join(shared) or 'None'}<br>
            🔴 Only in Q1: {', '.join(only_q1) or 'None'}<br>
            🔵 Only in Q2: {', '.join(only_q2) or 'None'}<br><br>
            ✅ The two questions convey similar intent or meaning even if phrased differently.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("❌ Not Duplicate")
            st.markdown(f"""
            <div class='output-card'>
            <h4>🧠 Why Not Duplicate:</h4>
            <b>🔍 Shared Keywords:</b> {', '.join(shared) or 'None'}<br>
            🔴 Only in Q1: {', '.join(only_q1) or 'None'}<br>
            🔵 Only in Q2: {', '.join(only_q2) or 'None'}<br><br>
            ❌ The two questions focus on different topics or contexts despite minor word overlap.
            </div>
            """, unsafe_allow_html=True)

