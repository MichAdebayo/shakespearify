import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from gtts import gTTS
import tempfile
from postprocessing import postprocess_shakespeare

st.set_page_config(page_title="Shakespeare Style", page_icon="🎭")
st.title("🎭 Shakespearean Translator")

image_placeholder = st.empty()
image_placeholder.image("img/shakespear.png", use_container_width=True)

@st.cache_resource
def load_models():
    fr_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    fr_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    es_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    es_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    shakespeare_tokenizer_online = AutoTokenizer.from_pretrained("Gorilla115/t5-shakespearify-lite")
    shakespeare_model_online = AutoModelForSeq2SeqLM.from_pretrained("Gorilla115/t5-shakespearify-lite")

    shakespeare_tokenizer_local = AutoTokenizer.from_pretrained("t5-shakespeare/checkpoint-34560/")
    shakespeare_model_local = AutoModelForSeq2SeqLM.from_pretrained("t5-shakespeare/checkpoint-34560/")

    return (fr_en_tokenizer, fr_en_model,
            es_en_tokenizer, es_en_model,
            shakespeare_tokenizer_online, shakespeare_model_online,
            shakespeare_tokenizer_local, shakespeare_model_local)

(fr_en_tokenizer, fr_en_model,
 es_en_tokenizer, es_en_model,
 shakespeare_tokenizer_online, shakespeare_model_online,
 shakespeare_tokenizer_local, shakespeare_model_local) = load_models()

# Language selector
language = st.selectbox("Select input language:", ("English", "French", "Spanish"))

# Default text based on language
if language == "French":
    default_text = "Je t'aime ma cherie"
elif language == "Spanish":
    default_text = "Te quiero mucho"
else:
    default_text = "I love you, my dear"

user_input = st.text_area("Enter text:", default_text)

# Model selector
model_choice = st.radio(
    "Choose Shakespeare model:",
    ("Online pretrained model", "My trained T5 Shakespeare model")
)

def translate(text, tokenizer, model, prefix=None):
    if prefix:
        text = f"{prefix}: {text}"
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if st.button("Translate to Shakespearean English"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Step 1: Translate to English if needed
        if language == "French":
            english_text = translate(user_input, fr_en_tokenizer, fr_en_model)
        elif language == "Spanish":
            english_text = translate(user_input, es_en_tokenizer, es_en_model)
        else:
            english_text = user_input

        # Step 2: Translate to Shakespearean English
        if model_choice == "Online pretrained model":
            shakespeare_text = translate(
                english_text,
                shakespeare_tokenizer_online,
                shakespeare_model_online,
                prefix="translate"
            )
        else:
            shakespeare_text = translate(
                english_text,
                shakespeare_tokenizer_local,
                shakespeare_model_local,
                prefix="translate"
            )
            shakespeare_text = postprocess_shakespeare(shakespeare_text)
        

        # Step 3: Show result
        st.markdown("### 🎭 Translated to Shakespearean English")
        st.success(shakespeare_text)

        # Step 4: Speech
        tts = gTTS(text=shakespeare_text, lang='en')
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts.save(fp.name)
            image_placeholder.image("img/shakespeartalking.gif", caption="Shakespeare Speaks", use_container_width=True)
            st.audio(fp.name, format='audio/mp3')

