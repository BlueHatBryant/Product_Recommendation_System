import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import html

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("sample-data.csv")

df = load_data()

# Preprocess descriptions (unescape HTML entities)
df["description"] = df["description"].apply(lambda x: html.unescape(x))

def get_short_description(text):
    sentences = text.split(". ")[:3]
    return ". ".join(sentences) + "."

# Compute similarity matrix
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Streamlit UI
st.title("Product Recommendation System")
st.write("By BlueHatBryant")
st.write("Select a product to see related recommendations.")

# Dropdown to select a product by ID
selected_id = st.selectbox("Choose a product id:", df["id"].tolist())

# Display selected product details
selected_product = df[df["id"] == selected_id].iloc[0]
st.subheader(selected_product["id"])
short_description = get_short_description(selected_product['description'])
st.markdown(f"*{short_description}*", unsafe_allow_html=True)

if st.button("Expand Description"):
    st.markdown(f"*{selected_product['description']}*", unsafe_allow_html=True)

# Get recommendations based on similarity
idx = df.index[df["id"] == selected_id][0]
similarities = list(enumerate(similarity_matrix[idx]))
similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[1:4]

# Display related products
st.subheader("Related Products:")
for i, score in similarities:
    related_product = df.iloc[i]
    related_short_description = get_short_description(related_product['description'])
    st.markdown(f"**{related_product['id']}** - {related_short_description}", unsafe_allow_html=True)
