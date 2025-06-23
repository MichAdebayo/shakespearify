import streamlit as st

st.set_page_config(page_title="How Transformers Work", page_icon="🧠")

page = st.sidebar.radio("Navigate", ["Overview", "Self-Attention", "T5-small", "Workflow"])

if page == "Overview":
    st.title("🧠 How Transformers Work: Overview")
    st.markdown("""
    Transformers are a type of deep learning model introduced in the paper *“Attention Is All You Need”* by Vaswani et al. (2017). They have revolutionized NLP tasks such as translation, summarization, and question answering.

    ---

    ### 🔄 Transformer Architecture Overview

    The Transformer consists of two main parts:

    1. **Encoder**: Converts the input sequence into a rich, contextualized representation.
    2. **Decoder**: Generates the output sequence based on the encoded representation.

    Key concepts include:
    - **Self-Attention**: Each token can "attend" to every other token in the sequence.
    - **Positional Encoding**: Adds order to the input sequence (since the architecture is order-agnostic).
    - **Multi-Head Attention**: Allows the model to attend to different parts of the sequence in parallel.
    """)

elif page == "Self-Attention":
    st.title("🧠 Deep Dive: What is Self-Attention?")
    st.markdown("""
    Self-attention allows each token in the input to focus on all other tokens, helping the model understand the context.

    #### How it Works:
    - Each token generates a **Query**, **Key**, and **Value** vector.
    - The attention mechanism compares queries to all keys to compute a relevance score.
    - These scores are normalized (using softmax) and used to weight the **Values**.
    - This weighted sum becomes the new representation of each token.

    #### Formula:
    ```
    Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V
    ```

    This process is repeated in parallel using multiple attention heads — known as **Multi-Head Attention** — allowing the model to capture different types of relationships in the sequence.
    """)

elif page == "T5-small":
    st.title("🧩 What is T5-small?")
    st.markdown("""
    #### In T5-small:
    - The encoder uses self-attention to model relationships **within the input**.
    - The decoder uses **masked self-attention** to prevent looking ahead.
    - The decoder also uses **cross-attention** to incorporate information from the encoder.
    - T5 reformulates every NLP task as text-to-text.
    - T5-small is ideal for lightweight inference and prototyping.
    **T5** stands for **Text-to-Text Transfer Transformer**. It was introduced by Google in the paper *“Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.”*

    **T5-small** is a compact version of T5 for quicker inference and lower memory usage.

    | Feature | Value |
    |--------|-------|
    | Model Name | `t5-small` |
    | Parameters | ~60 million |
    | Type | Encoder-Decoder |
    | Input Format | Text-to-Text (e.g., `translate English to Shakespearean: <text>`) |
    """)

elif page == "Workflow":
    st.title("🛠 Example Workflow with T5")
    st.markdown("""
    1. **Input**:
        - A plain text command like `summarize: The quick brown fox jumps over the lazy dog.`

    2. **Tokenization**:
        - The text is tokenized into IDs using a SentencePiece tokenizer.

    3. **Encoding**:
        - The encoder reads the input and builds a contextual representation using self-attention.

    4. **Decoding**:
        - The decoder generates the output token-by-token using:
            - Masked self-attention
            - Encoder-decoder attention

    5. **Output**:
        - A generated sequence like `A quick brown fox jumps a lazy dog.`
    """)
