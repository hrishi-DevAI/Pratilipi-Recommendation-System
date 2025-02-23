import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder



@st.cache_resource
def load_ncf_model():
    # Adjust path if needed
    return tf.keras.models.load_model("Models/pratilipi_recommendation_model.keras")

@st.cache_data
def load_label_encoders():
    with open("Encoders/user_encoder.pkl", "rb") as f_user:
        user_enc = pickle.load(f_user)
    with open("Encoders/pratilipi_encoder.pkl", "rb") as f_item:
        pratilipi_enc = pickle.load(f_item)
    return user_enc, pratilipi_enc



def recommend_for_user(
    model,
    user_id_raw,
    user_encoder,
    pratilipi_encoder,
    top_n=5
):
    """
    Predict 'read probability' for all pratilipis given a user, 
    then return the top-N pratilipi IDs.
    """
    # Encode the user
    user_id_encoded = user_encoder.transform([user_id_raw])[0]

    # Encode all pratilipis
    all_pratilipis = np.arange(len(pratilipi_encoder.classes_))  # 0..num_pratilipis-1
    
    # Build model inputs
    user_array = np.full(shape=len(all_pratilipis), fill_value=user_id_encoded)
    pratilipi_array = all_pratilipis

    # Predict
    preds = model.predict([user_array, pratilipi_array], batch_size=1024).flatten()

    # Get top-N
    top_indices = np.argsort(preds)[-top_n:][::-1]  # sort ascending, take last N, reverse
    # Decode back to raw IDs
    recommended_raw_ids = pratilipi_encoder.inverse_transform(top_indices)
    return recommended_raw_ids


def recommend_similar_items(
    model,
    pratilipi_id_raw,
    pratilipi_encoder,
    top_n=5
):
    """
    Item-based approach: find the items most similar to a given item
    based on the NCF model's learned item embeddings.
    """
    # Extract item embeddings from the model
    pratilipi_embedding_layer = model.get_layer("pratilipi_embedding")
    item_embeddings = pratilipi_embedding_layer.get_weights()[0]  # shape: (num_items, embedding_dim)

    # Encode the raw ID
    pratilipi_id_encoded = pratilipi_encoder.transform([pratilipi_id_raw])[0]

    # Cosine similarity with every item
    target_vec = item_embeddings[pratilipi_id_encoded]
    dot_products = np.dot(item_embeddings, target_vec)
    norms = np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(target_vec)
    cosine_sim = dot_products / (norms + 1e-10)

    # Sort descending by similarity
    most_sim_indices = np.argsort(cosine_sim)[::-1]

    # Exclude the item itself
    most_sim_indices = most_sim_indices[most_sim_indices != pratilipi_id_encoded]

    # Take top-N
    top_n_indices = most_sim_indices[:top_n]

    # Decode
    recommended_raw_ids = pratilipi_encoder.inverse_transform(top_n_indices)
    return recommended_raw_ids


def main():
    st.title("NCF Recommendation Demo")

    # Load resources
    model = load_ncf_model()
    user_encoder, pratilipi_encoder = load_label_encoders()

    # Let user pick which method
    st.subheader("How would you like to get recommendations?")
    rec_choice = st.radio(
        label="Select a method:",
        options=["By User ID", "By Pratilipi ID"]
    )

    if rec_choice == "By User ID":

        st.write("Enter a user ID to get top-N recommended pratilipis for that user.")
        
        user_id_input = st.text_input("User ID (raw, e.g. 5506791996648218):", "")
        topn_value = st.number_input("Number of recommendations (top-N):", min_value=1, max_value=20, value=5)

        if st.button("Get User-Based Recommendations"):
            if user_id_input.strip():
                try:
                    user_id_raw = str(user_id_input)
                    recs = recommend_for_user(
                        model=model,
                        user_id_raw=user_id_raw,
                        user_encoder=user_encoder,
                        pratilipi_encoder=pratilipi_encoder,
                        top_n=topn_value
                    )
                    st.success(f"Recommended for user {user_id_raw}:")
                    for r in recs:
                        st.write(f"- {r}")
                except ValueError:
                    st.error("Please enter a valid integer for user ID.")
            else:
                st.warning("Please enter a user ID.")

    else:

        st.write("Enter a pratilipi ID you like to get similar pratilipis.")

        pratilipi_id_input = st.text_input("Pratilipi ID (raw):", "")
        topn_value = st.number_input("Number of similar items (top-N):", min_value=1, max_value=20, value=5)

        if st.button("Get Similar Pratilipis"):
            if pratilipi_id_input.strip():
                try:
                    pratilipi_id_raw = str(pratilipi_id_input)
                    recs = recommend_similar_items(
                        model=model,
                        pratilipi_id_raw=pratilipi_id_raw,
                        pratilipi_encoder=pratilipi_encoder,
                        top_n=topn_value
                    )
                    st.success(f"Similar pratilipis to {pratilipi_id_raw}:")
                    for r in recs:
                        st.write(f"- {r}")
                except ValueError:
                    st.error("Please enter a valid integer for pratilipi ID.")
            else:
                st.warning("Please enter a pratilipi ID.")

if __name__ == "__main__":
    main()

