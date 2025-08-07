import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

# Load model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("preshnkelsie/igbo-ner")
    model = AutoModelForTokenClassification.from_pretrained("preshnkelsie/igbo-ner")
    return tokenizer, model

tokenizer, model = load_model()
label_list = model.config.id2label

st.title("Igbo Named Entity Recognition (NER)")
st.write("Enter Igbo text below to extract entities using your fine-tuned model.")

user_input = st.text_area("Igbo Text", height=150)

def get_predictions(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)
    probs = torch.nn.functional.softmax(outputs.logits, dim=2)

    results = []
    for token_idx, token_id in enumerate(tokens["input_ids"][0]):
        token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
        label_id = predictions[0][token_idx].item()
        label = label_list[label_id]
        score = probs[0][token_idx][label_id].item()

        # Skip special tokens
        if token_str not in tokenizer.all_special_tokens:
            results.append((token_str, label, round(score, 2)))
    return results

if st.button("Extract Entities"):
    if user_input.strip():
        results = get_predictions(user_input)
        st.write("### BIO-tagged Output:")
        for token, label, score in results:
            st.write(f"**{token}** → {label} (confidence: {score})")
    else:
        st.warning("Please enter some text.")