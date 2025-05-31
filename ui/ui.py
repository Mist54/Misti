import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from nlp.session import nlp_userInput
from nlp.nlp_engine import process_input

# Initialize NLP session
if "nlp_session" not in st.session_state:
    st.session_state["nlp_session"] = nlp_userInput()

st.title("NLP Input Processor")

# Text input area
user_input = st.text_input("Enter your message:")

# Buttons in a single row
col1, col2, col3 = st.columns(3)
with col1:
    submit = st.button("Submit")
with col2:
    view = st.button("View Session")  # Retained but optional now
with col3:
    st.write("")  # placeholder for alignment

# Sidebar for session data and clear
with st.sidebar:
    st.subheader("User Session Data")
    st.write(st.session_state["nlp_session"].get())
    if st.button("Clear Session", key="clear_sidebar"):
        st.session_state["nlp_session"].clear()
        st.success("Session cleared.")

# Submit logic (process user input)
if submit and user_input:
    st.session_state["nlp_session"].set(user_input)
    full_context = " ".join(st.session_state["nlp_session"].get())
    result = process_input(full_context)
    st.subheader("Extracted Info")
    st.json(result)

# Optional view logic (already shown in sidebar)
if view:
    st.subheader("Current Session Data")
    st.write(st.session_state["nlp_session"].get())
