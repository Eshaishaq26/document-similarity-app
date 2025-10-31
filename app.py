import streamlit as st
import PyPDF2
import re
import pandas as pd
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import io


def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess(text):
    """Lowercase, remove punctuation, and split into word set."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return set(text.split())

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two word sets."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

# ------------------ Streamlit UI ------------------

st.title("Plagiarism Checker using Jaccard Similarity")
st.write("Upload 2 or more PDF documents to check textual similarity.")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    pdf_texts = {}
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        pdf_texts[file.name] = preprocess(text)

    # Compute Jaccard similarity for all unique pairs
    results = []
    for (name1, text1), (name2, text2) in combinations(pdf_texts.items(), 2):
        sim = round(jaccard_similarity(text1, text2) * 100, 2)
        results.append((name1, name2, sim))

    df = pd.DataFrame(results, columns=["Document 1", "Document 2", "Similarity (%)"])
    st.subheader("📋 Similarity Results")
    st.dataframe(df)

    # --------- Visualization 1: Bar Chart ---------
    st.subheader("Bar Chart of Similarity")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="Similarity (%)", y="Document 1", hue="Document 2", palette="Blues_d", ax=ax1)
    plt.legend(title="Compared With", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig1)

    # --------- Visualization 2: Heatmap ---------
    st.subheader(" Heatmap of Similarities")

    # Create symmetric matrix
    all_docs = sorted(set(df["Document 1"]).union(df["Document 2"]))
    matrix = pd.DataFrame(0, index=all_docs, columns=all_docs)

    for _, row in df.iterrows():
        d1, d2, sim = row["Document 1"], row["Document 2"], row["Similarity (%)"]
        matrix.loc[d1, d2] = sim
        matrix.loc[d2, d1] = sim
        matrix.loc[d1, d1] = 100

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Similarity %'}, ax=ax2)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig2)

    st.success("Analysis complete!")

else:
    st.info("Please upload at least two PDF files.")