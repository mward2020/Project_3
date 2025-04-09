import streamlit as st

st.set_page_config(
    page_title="Chatbot Playground",
    page_icon="🤖",
)

st.title("🤖 Chatbot Playground")
st.write("Welcome to the Chatbot Playground!")

st.markdown("""
Use the sidebar to select a chatbot:

- 🧘‍♀️ **Good Bot**: Friendly, uplifting advice
- 😈 **Bad Bot**: Malicious advice for demonstration only

---

This project shows how AI behaviors can be tuned by prompt design and model control.
""")