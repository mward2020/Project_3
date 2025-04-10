
# 🧠 Mental Health Chatbot

An AI-powered chatbot designed to support mental wellness by providing empathetic, multi-label emotional responses based on user input. Developed using state-of-the-art NLP models and deployed via a clean Streamlit interface.

![Image](https://github.com/user-attachments/assets/e6db438c-eded-42ff-a893-3dece893564e)


## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data](#-data)
- [Model Development](#-model-development)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Full Technical Summary](#-full-technical-summary)
- [Future Enhancements](#-future-enhancements)
- [Project Requirements](#-project-requirements)
- [Resources](#-resources)

---

## 🧠 Project Overview

This repository contains the complete development of a Mental Health Chatbot. The chatbot provides real-time, accessible, and private support to individuals facing mental health challenges. It serves as a tool for emotional validation, self-help resources, and guidance toward professional care.
This notebook implements a robust, emotion-aware chatbot system that combines emotional classification with response generation using a RoBERTa + T5 hybrid architecture. The system is trained on multiple datasets, supports real-time chat via Gradio, and saves all models and metadata for reproducibility.

---

## 🔧 Installation

```bash
git clone https://github.com/your-repo/Mental-Health-Chatbot.git
cd Mental-Health-Chatbot
pip install -r requirements.txt
```

---

## 🚀 Usage

To run the chatbot locally:
```bash
streamlit run app.py
```

This launches the web app with a chat interface that supports emotional understanding.

---

## 📂 Data

Trained on mental health Q&A datasets from Kaggle and Hugging Face, preprocessed and labeled with multi-label binarization and tokenized with Hugging Face tokenizers.

---

## 🤖 Model Development

### Initial Model
- DistilBERT baseline
- Limitations in multi-emotion understanding

### Enhanced Model
- RoBERTa (emotion classification)
- Two T5 models (Chat-style & QA-style responses)
- BCEWithLogitsLoss + custom routing logic

---

## 📊 Evaluation

- Precision, Recall, F1-score
- Visual and sample-based assessment
- Reduced hallucinations

---

## 🧪 Results

- 10–15% improvement in multi-label accuracy
- Reliable tone detection and emotional coherence
- Lightweight deployment with Streamlit

---

## 📘 Full Technical Summary

This system combines emotional classification with response generation using a RoBERTa + T5 hybrid. It loads and preprocesses mental health datasets, performs multi-label binarization for emotional tagging, and routes user queries based on emotional detection.

### Highlights:
- Modular: separate RoBERTa + 2x T5s
- Routed inference logic based on emotion scores
- Saves all models/tokenizers with `.pt` metadata
- Integrated with a Gradio UI for real-time chat

Refer to [summary.txt](summary.txt) for the full breakdown.

---

## 🚀 Future Enhancements

- Multilingual capability
- Voice input integration
- Therapist handoff system
- Fine-tuning via live feedback

---

## 📌 Project Requirements

- ✅ Model Implementation
- ✅ Model Optimization
- ✅ GitHub Documentation
- ✅ Presentation Delivery

---

## 📎 Resources

- [Hugging Face Docs](https://huggingface.co)
- [Streamlit](https://docs.streamlit.io)
- [Gradio](https://www.gradio.app)
- [ChatGPT](https://www.chatgpt.com)
- [Kaggle](https://www.Kaggle.com)

---

_This tool supports early emotional help but does not replace professional therapy._
