import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)

class PipelineConfig:
    def __init__(self, emo_path, qa_path, resp_path, labels):
        self.emo_path = emo_path
        self.qa_path = qa_path
        self.resp_path = resp_path
        self.labels = labels

class MentalHealthChatbotPipeline:
    def __init__(self, emo_model, emo_tokenizer, qa_model, qa_tokenizer, response_model, response_tokenizer, labels):
        self.emo_model = emo_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.emo_tokenizer = emo_tokenizer
        self.qa_model = qa_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.qa_tokenizer = qa_tokenizer
        self.response_model = response_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.response_tokenizer = response_tokenizer
        self.labels = labels

    def analyze(self, text):
        device = next(self.emo_model.parameters()).device

        emo_inputs = self.emo_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = self.emo_model(**emo_inputs).logits
        probs = torch.sigmoid(logits).squeeze().cpu().tolist()
        emotions = [self.labels[i] for i, p in enumerate(probs) if p > 0.5]

        qa_input = f"question: {text}"
        qa_inputs = self.qa_tokenizer(qa_input, return_tensors="pt").to(device)
        with torch.no_grad():
            qa_ids = self.qa_model.generate(qa_inputs["input_ids"], max_length=128)
        answer = self.qa_tokenizer.decode(qa_ids[0], skip_special_tokens=True)

        prompt = f"question: {text} context: {answer} emotions: {', '.join(emotions)}"
        response_inputs = self.response_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            response_ids = self.response_model.generate(response_inputs["input_ids"], max_length=128)
        response = self.response_tokenizer.decode(response_ids[0], skip_special_tokens=True)

        return {
            "emotions": emotions,
            "answer": answer,
            "response": response
        }

# Load pointer
config_path = "./saved_models/final_combined/pipeline_config.pt"
config = torch.load(config_path)

# Load models and tokenizers
emo_model = AutoModelForSequenceClassification.from_pretrained(config.emo_path)
emo_tokenizer = AutoTokenizer.from_pretrained(config.emo_path)

qa_model = T5ForConditionalGeneration.from_pretrained(config.qa_path)
qa_tokenizer = T5Tokenizer.from_pretrained(config.qa_path)

resp_model = T5ForConditionalGeneration.from_pretrained(config.resp_path)
resp_tokenizer = T5Tokenizer.from_pretrained(config.resp_path)

# Create the full pipeline
chatbot = MentalHealthChatbotPipeline(
    emo_model=emo_model,
    emo_tokenizer=emo_tokenizer,
    qa_model=qa_model,
    qa_tokenizer=qa_tokenizer,
    response_model=resp_model,
    response_tokenizer=resp_tokenizer,
    labels=config.labels
)

# Run a sample
sample_text = "I'm feeling overwhelmed and disconnected lately."
result = chatbot.analyze(sample_text)

print("\nDetected Emotions:", result["emotions"])
print("QA Model Answer:", result["answer"])
print("Final Chatbot Response:", result["response"])
