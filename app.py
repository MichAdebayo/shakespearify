import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Streamlit config
st.set_page_config(page_title="Shakespearify + FastSpeech2 TTS", page_icon="🎭")

st.title("🎭 Shakespearify Modern English + Read Aloud")
st.markdown("Convert modern English to Shakespearean style and hear it spoken aloud.")

# Load Shakespeare model once
@st.cache_resource
def load_shakespeare_model():
    tokenizer = AutoTokenizer.from_pretrained("Gorilla115/t5-shakespearify-lite")
    model = AutoModelForSeq2SeqLM.from_pretrained("Gorilla115/t5-shakespearify-lite")
    return tokenizer, model

# Load TTS pipeline once
@st.cache_resource
def load_tts_pipeline():
    return pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")

tokenizer, model = load_shakespeare_model()
tts = load_tts_pipeline()

user_input = st.text_area("Enter modern English text:", "To be or not to be, that is the question.")

if st.button("Shakespearify and Read Aloud"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Convert to Shakespearean
        input_text = f"translate English to Shakespearean: {user_input}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
        shakespeare_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.markdown("### 📝 Shakespearean Style Output")
        st.success(shakespeare_text)

        # Generate audio from Shakespearean text
        audio_output = tts(shakespeare_text)

        # The pipeline returns dict with "wav" key containing numpy array of audio
        audio_array = audio_output["wav"]
        # Save audio to temp file to play in Streamlit
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_array, samplerate=22050)  # default sample rate for this model
            temp_audio_path = f.name

        audio_bytes = open(temp_audio_path, "rb").read()
        st.audio(audio_bytes, format="audio/wav")

        import os
        os.unlink(temp_audio_path)
