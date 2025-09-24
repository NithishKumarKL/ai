# app/streamlit_app.py
import streamlit as st
import requests
import os

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="Insurance FAQ RAG Assistant", layout="centered")
st.title("Insurance Policy FAQ Assistant")

question = st.text_input("Ask your policy question here:", "")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving answers..."):
        resp = requests.post(API_URL, json={"question": question})
    if resp.status_code != 200:
        st.error(f"Error: {resp.text}")
    else:
        data = resp.json()
        st.subheader("Answer")
        st.write(data["answer"])
        st.markdown("**Citations:**")
        for s in data["sources"]:
            st.write(f"- [FAQ {s}]")
        st.markdown("---")
        st.subheader("Retrieved contexts")
        for h in data["hits"]:
            st.markdown(f"**[FAQ {h['faq_id']}]** — {h['question']}")
            st.write(h['document'])
            st.write(f"_distance_: {h['distance']:.4f}")
