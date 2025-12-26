import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Central Bank AI Watcher",
    page_icon="🦅",
    layout="centered"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """
    Loads FinBERT model and tokenizer. Cached to prevent reloading on every interaction.
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    # Use CPU for deployment stability on standard instances
    device = torch.device("cpu")
    model.to(device)
    return tokenizer, model, device

# Initialize model
with st.spinner('Initializing AI Model (FinBERT)...'):
    tokenizer, model, device = load_model()

# --- INFERENCE LOGIC ---
def predict_sentiment(text):
    """
    Analyzes text sentiment using FinBERT. 
    Handles tokenization and chunking for long texts (>512 tokens).
    Returns: Mean probability array [Positive, Negative, Neutral]
    """
    if not text or len(text.split()) < 5:
        return None
    
    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')
    input_id_chunks = tokens['input_ids'][0].split(500)
    mask_chunks = tokens['attention_mask'][0].split(500)
    
    chunk_probs = []

    # Process chunks (sliding window approach not strictly necessary here, simple split suffices)
    for i in range(len(input_id_chunks)):
        input_ids = torch.cat([
            torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
        ]).unsqueeze(0).to(device)
        
        attention_mask = torch.cat([
            torch.tensor([1]), mask_chunks[i], torch.tensor([1])
        ]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            chunk_probs.append(probs.cpu().numpy()[0])
            
    return np.mean(chunk_probs, axis=0)

# --- USER INTERFACE ---

st.title("🦅 Central Bank AI Watcher")
st.markdown("""
This tool utilizes **FinBERT** (Financial Bidirectional Encoder Representations from Transformers) to analyze the sentiment of central bank statements and financial news.
""")

input_text = st.text_area("Paste statement or article below:", height=250)

if st.button("Analyze Sentiment"):
    if input_text:
        with st.spinner('Processing...'):
            probs = predict_sentiment(input_text)

            if probs is not None:
                # FinBERT Output Order: [Positive, Negative, Neutral]
                pos_score, neg_score, neu_score = probs[0], probs[1], probs[2]
                
                # Composite Score: Negative vs Positive
                hawk_score = neg_score - pos_score

                st.success("Analysis Complete")
                st.divider()

                # Metrics Display
                col1, col2, col3 = st.columns(3)
                col1.metric("Dovish (Positive)", f"{pos_score:.1%}")
                col2.metric("Neutral", f"{neu_score:.1%}")
                col3.metric("Hawkish (Negative)", f"{neg_score:.1%}")
                
                st.subheader("Verdict")

                # Interpretation Logic
                if hawk_score > 0.1:
                    st.error(f"⚠️ **HAWKISH BIAS**: The model detects concern regarding inflation or potential tightening. Score: +{hawk_score:.2f}")
                elif hawk_score < -0.1:
                    st.success(f"🕊️ **DOVISH BIAS**: The model detects an accommodative or reassuring tone. Score: {hawk_score:.2f}")
                else:
                    st.info(f"⚖️ **NEUTRAL**: The text is balanced or purely technical. Score: {hawk_score:.2f}")

                # Visualization
                st.write("Confidence Breakdown:")
                st.caption("Dovish / Positive 🟢")
                st.progress(float(pos_score))
                st.caption("Hawkish / Negative 🔴")
                st.progress(float(neg_score))
            else:
                st.warning("Input text is too short for meaningful analysis.")
    else:
        st.warning("Please enter text to analyze.")