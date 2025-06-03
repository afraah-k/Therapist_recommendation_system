import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import torch

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv("Therapist_Tagging_Final_Dataset.csv")

    text_columns = [
        'Response Style', 'Emotional Attunement', 'Modality Leaning', 'Handling of Resistance',
        'Crisis Response', 'Feedback Openness', 'Power Sharing', 'Emotional Language Depth',
        'Therapeutic Warmth', 'Safety vs Challenge', 'Specialties'
    ]

    df['Combined_Description'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    return df

df = load_data()

# Load model and embeddings
@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = model.encode(df['Combined_Description'].tolist(), convert_to_tensor=True)
    return model, corpus_embeddings

model, corpus_embeddings = load_model_and_embeddings(df)

# Streamlit Sidebar - Inputs
st.sidebar.title("🧠 Therapist Matcher")
st.sidebar.markdown("Tell us your concern and what kind of therapist you're looking for.")

user_problem = st.sidebar.text_area("💬 What's troubling you?", height=100, placeholder="e.g., I have anxiety and panic attacks")
user_preference = st.sidebar.text_area("🌿 Describe your ideal therapist", height=100, placeholder="e.g., Someone who is calm and uses CBT")

# Optional Filter: Modality
available_modalities = df['Modality Leaning'].dropna().unique().tolist()
modality_filter = st.sidebar.selectbox("🧩 Preferred Modality (optional)", ["Any"] + sorted(available_modalities))

# Submit button
submit = st.sidebar.button("🔍 Find Therapists")

# Matching logic function
def get_top_matches_semantic(query, df_filtered, model, corpus_embeddings_filtered, top_n=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings_filtered)[0]
    top_indices = similarity_scores.argsort(descending=True)[:top_n]
    return df_filtered.iloc[[i.item() for i in top_indices]], similarity_scores[top_indices]

# Main display logic
if submit:
    if user_problem.strip() == "" or user_preference.strip() == "":
        st.warning("Please fill out both the problem and therapist preference fields.")
    else:
        user_input = user_problem + ". " + user_preference

        # Apply modality filter if selected
        if modality_filter != "Any":
            df_filtered = df[df['Modality Leaning'] == modality_filter]
            if df_filtered.empty:
                st.error("No therapists match the selected modality.")
            else:
                corpus_embeddings_filtered = model.encode(df_filtered['Combined_Description'].tolist(), convert_to_tensor=True)
                top_matches, scores = get_top_matches_semantic(user_input, df_filtered, model, corpus_embeddings_filtered)
        else:
            df_filtered = df
            corpus_embeddings_filtered = corpus_embeddings
            top_matches, scores = get_top_matches_semantic(user_input, df_filtered, model, corpus_embeddings_filtered)

        if not top_matches.empty:
            st.header("🎯 Top Therapist Matches")
            for i, (index, row) in enumerate(top_matches.iterrows()):
                st.subheader(f"{i+1}. {row['Therapist Name']}")
                st.write(f"**Specialties:** {row['Specialties']}")
                st.write(f"**Emotional Language Depth:** {row['Emotional Language Depth']}")
                st.write(f"**Match Score:** {scores[i].item():.4f}")
                st.markdown("---")
        else:
            st.warning("No matches found based on your input.")

# Data Insights
st.sidebar.markdown("### 📊 Data Insights")

if st.sidebar.checkbox("Show Specialties Distribution"):
    st.header("Top Specialties in Dataset")
    top_specialties = df["Specialties"].value_counts().head(10)
    fig, ax = plt.subplots()
    top_specialties.plot(kind='barh', ax=ax, color='skyblue')
    st.pyplot(fig)

if st.sidebar.checkbox("Show Modality Usage"):
    if "Modality Leaning" in df.columns:
        st.header("Modality Leaning Distribution")
        top_modality = df["Modality Leaning"].value_counts().head(10)
        fig, ax = plt.subplots()
        top_modality.plot(kind='barh', ax=ax, color='lightgreen')
        st.pyplot(fig)





