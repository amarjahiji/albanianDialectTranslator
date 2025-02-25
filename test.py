import streamlit as st
import torch
from model import model, src_vocab, tgt_vocab, translate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    from dataset import train_pairs
except ImportError:
    train_pairs = []
    st.warning("Test dataset not found. Evaluation features will be limited.")

# Load the trained model
try:
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    st.error("Model file not found. Please train the model first.")

# Evaluation functions
def calculate_bleu(reference, candidate):
    # Simple word splitting as tokenization
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    # Using smoothing to handle edge cases
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

def calculate_cer(reference, candidate):
    return Levenshtein.distance(candidate, reference) / max(len(reference), 1)

def calculate_wer(reference, candidate):
    ref_words = reference.split()
    cand_words = candidate.split()
    distance = Levenshtein.distance(cand_words, ref_words)
    return distance / max(len(ref_words), 1)

def evaluate_on_test_set(model, train_pairs, src_vocab, tgt_vocab, num_samples=100):
    results = []
    
    # Limit evaluation to a reasonable number of samples
    samples = train_pairs[:min(num_samples, len(train_pairs))]
    
    for i, (src_text, tgt_text) in enumerate(samples):
        # Get model translation
        pred_text = translate(model, src_text, src_vocab, tgt_vocab)
        
        # Calculate metrics
        bleu = calculate_bleu(tgt_text, pred_text)
        cer = calculate_cer(tgt_text, pred_text)
        wer = calculate_wer(tgt_text, pred_text)
        
        results.append({
            'id': i+1,
            'source': src_text,
            'reference': tgt_text,
            'prediction': pred_text,
            'bleu': bleu,
            'cer': cer,
            'wer': wer
        })
    
    return pd.DataFrame(results)

# Main Streamlit App
st.title("Albanian Dialect Translator")
st.write("Enter a sentence in Gheg dialect to translate to Standard Albanian")

# Create tabs for translation and evaluation
tab1, tab2 = st.tabs(["Translation", "Model Evaluation"])

with tab1:
    # Input text box for user input
    user_input = st.text_input("Enter text to translate:")
    
    # Reference text for comparison (optional)
    reference_input = st.text_input("Reference (optional - if you know the standard Albanian):")

    # Add a translate button
    if st.button("Translate"):
        if user_input:
            with st.spinner('Translating...'):
                translation = translate(model, user_input, src_vocab, tgt_vocab)
            
            # Display results
            st.subheader("Translation:")
            st.success(translation)
            
            # Show additional information
            st.subheader("Translation Details:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Text:**")
                st.write(user_input)
            with col2:
                st.write("**Translated Text:**")
                st.write(translation)
            
            # If reference is provided, calculate metrics
            if reference_input:
                st.subheader("Evaluation Metrics:")
                bleu = calculate_bleu(reference_input, translation)
                cer = calculate_cer(reference_input, translation)
                wer = calculate_wer(reference_input, translation)
                
                metrics_df = pd.DataFrame({
                    'Metric': ['BLEU Score', 'Character Error Rate (CER)', 'Word Error Rate (WER)'],
                    'Value': [f"{bleu:.4f}", f"{cer:.4f}", f"{wer:.4f}"],
                    'Interpretation': [
                        "Higher is better (0-1 scale)",
                        "Lower is better (0-1 scale)",
                        "Lower is better (0-1 scale)"
                    ]
                })
                st.table(metrics_df)
        else:
            st.warning("Please enter some text to translate")

with tab2:
    st.header("Model Evaluation Dashboard")
    
    if len(train_pairs) > 0:
        if st.button("Run Evaluation on Test Set"):
            with st.spinner("Evaluating model on test set..."):
                results_df = evaluate_on_test_set(model, train_pairs, src_vocab, tgt_vocab)
                
                # Display aggregate metrics
                st.subheader("Aggregate Metrics")
                agg_metrics = pd.DataFrame({
                    'Metric': ['Avg BLEU Score', 'Avg Character Error Rate', 'Avg Word Error Rate'],
                    'Value': [
                        f"{results_df['bleu'].mean():.4f}",
                        f"{results_df['cer'].mean():.4f}",
                        f"{results_df['wer'].mean():.4f}"
                    ]
                })
                st.table(agg_metrics)
                
                # Plot distribution of metrics
                st.subheader("Metrics Distribution")
                metrics_plot = pd.DataFrame({
                    'BLEU Score': results_df['bleu'],
                    'Character Error Rate': results_df['cer'],
                    'Word Error Rate': results_df['wer']
                })
                st.line_chart(metrics_plot)
                
                # Show example translations
                st.subheader("Sample Translations")
                sample_df = results_df[['source', 'reference', 'prediction', 'bleu', 'cer']].head(10)
                st.dataframe(sample_df)
                
                # Allow downloading full results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Full Evaluation Results",
                    csv,
                    "dialect_translation_evaluation.csv",
                    "text/csv",
                    key='download-csv'
                )
    else:
        st.info("Test dataset not available. Please make sure train_pairs is imported correctly.")
        
    # Manual evaluation form
    st.subheader("Manual Evaluation")
    st.write("You can evaluate specific translation examples here:")
    
    manual_source = st.text_area("Source Text (Gheg dialect):")
    manual_reference = st.text_area("Reference Text (Standard Albanian):")
    
    if st.button("Evaluate Example"):
        if manual_source and manual_reference:
            with st.spinner("Translating and evaluating..."):
                manual_prediction = translate(model, manual_source, src_vocab, tgt_vocab)
                
                # Calculate metrics
                manual_bleu = calculate_bleu(manual_reference, manual_prediction)
                manual_cer = calculate_cer(manual_reference, manual_prediction)
                manual_wer = calculate_wer(manual_reference, manual_prediction)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Translation Result:**")
                    st.write(manual_prediction)
                with col2:
                    st.write("**Evaluation Metrics:**")
                    st.write(f"BLEU Score: {manual_bleu:.4f}")
                    st.write(f"Character Error Rate: {manual_cer:.4f}")
                    st.write(f"Word Error Rate: {manual_wer:.4f}")
        else:
            st.warning("Both source and reference texts are required for evaluation")

# Sidebar content
st.sidebar.header("About")
st.sidebar.info(
    "This application translates Gheg dialect Albanian to Standard Albanian "
    "using a Transformer neural network model trained on paired examples."
)

# Add metrics/stats
st.sidebar.header("Model Information")
st.sidebar.markdown(f"""
- Vocabulary size (source): {len(src_vocab.chars)}
- Vocabulary size (target): {len(tgt_vocab.chars)}
- Model type: Transformer
- Embedding dimension: {model.d_model}
- Attention heads: {model.decoder.layers[0].multihead_attn.num_heads}
""")

# Add explanation of metrics
st.sidebar.header("Evaluation Metrics")
st.sidebar.markdown("""
**BLEU Score**
- Measures n-gram overlap with reference
- Range: 0-1 (higher is better)
- Industry standard for MT evaluation

**Character Error Rate (CER)**
- Edit distance at character level
- Range: 0-1 (lower is better)
- Important for closely related dialects

**Word Error Rate (WER)**
- Edit distance at word level
- Range: 0-1 (lower is better)
- Measures semantic accuracy
""")