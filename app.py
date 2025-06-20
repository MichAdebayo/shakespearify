import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set Streamlit page config
st.set_page_config(page_title="Shakespeare Style", page_icon="🎭")

st.title("🎭 Shakespearean Translator")

st.image("shakespear.png", caption="William Shakespeare", use_container_width=True)

# Load models (cache them)
@st.cache_resource
def load_models():
    # French to English model and tokenizer
    fr_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    fr_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    # English to Shakespeare model and tokenizer
    shakespeare_tokenizer = AutoTokenizer.from_pretrained("Gorilla115/t5-shakespearify-lite")
    shakespeare_model = AutoModelForSeq2SeqLM.from_pretrained("Gorilla115/t5-shakespearify-lite")

    return fr_en_tokenizer, fr_en_model, shakespeare_tokenizer, shakespeare_model

fr_en_tokenizer, fr_en_model, shakespeare_tokenizer, shakespeare_model = load_models()

# User input: French text
user_input = st.text_area("Enter text:", "Je t'aime ma cherie")

def translate(text, tokenizer, model, prefix=None):
    # Prepare input with optional prefix (for models like T5)
    if prefix:
        text = f"{prefix}: {text}"
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if st.button("Translate to Shakespearean English"):
    if user_input.strip() == "":
        st.warning("Please enter some French text.")
    else:
        # Step 1: French -> English
        english_text = translate(user_input, fr_en_tokenizer, fr_en_model)

        # Step 2: English -> Shakespearean English
        shakespeare_text = translate(
            english_text,
            shakespeare_tokenizer,
            shakespeare_model,
            prefix="translate:"
        )
        st.markdown("### 🎭 Translated to Shakespearean English")
        st.success(shakespeare_text)
