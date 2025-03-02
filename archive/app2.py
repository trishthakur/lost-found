import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load the CLIP model and processor (only once at startup)
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", padding=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

model, processor, device = load_model()

# Paths
image_folder = "lost_items"
os.makedirs(image_folder, exist_ok=True)
excel_file = "lost_items.csv"  # Using CSV instead of Excel to avoid permission issues
locations_file = "locations.xlsx"

# Initialize CSV file if not present
if not Path(excel_file).exists():
    pd.DataFrame(columns=["Image Name", "Location", "Timestamp", "Embedding", "Description", "Status"]).to_csv(excel_file, index=False)

# Load location data
locations_df = pd.read_excel(locations_file)
campus_options = locations_df['Campus'].unique()

def encode_text_and_image(text=None, image_path=None, image=None):
    try:
        if text is None:
            text = ""
        if image is None:
            image = Image.new('RGB', (224, 224))
        
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            text_embedding = outputs.text_embeds
            image_embedding = outputs.image_embeds

        return text_embedding, image_embedding
    except Exception as e:
        st.error(f"Error processing input: {e}")
        return None, None

def find_best_match(text_query=None, image_query=None, image_folder=image_folder):
    best_match = None
    best_score = -1
    df = pd.read_csv(excel_file)

    if 'Embedding' not in df.columns:
        st.error("'Embedding' column is missing from the CSV file.")
        return None, -1

    if text_query or image_query:
        query_text_emb, query_img_emb = encode_text_and_image(text=text_query, image=image_query)
        
        for _, row in df.iterrows():
            if pd.notnull(row['Embedding']):
                try:
                    stored_embedding = torch.tensor(eval(row['Embedding']))
                    
                    score = 0
                    if query_text_emb is not None and stored_embedding is not None:
                        score += torch.cosine_similarity(query_text_emb, stored_embedding).item()
                    if query_img_emb is not None and stored_embedding is not None:
                        score += torch.cosine_similarity(query_img_emb, stored_embedding).item()
                    
                    if score > best_score:
                        best_score = score
                        best_match = row['Image Name']
                except (SyntaxError, NameError, TypeError):
                    st.warning(f"Invalid embedding data for {row['Image Name']}. Skipping.")
                    continue

    return best_match, best_score

# Streamlit UI
st.title("Lost and Found Item Matcher")

df = pd.read_csv(excel_file)

option = st.radio("Choose an option:", ("Report a Lost Item", "Find a Lost Item"))

if option == "Report a Lost Item":
    uploaded_file = st.file_uploader("Upload an image of the lost item", type=["png", "jpg", "jpeg"])
    
    campus = st.selectbox("Select Campus", campus_options)
    buildings = locations_df[locations_df['Campus'] == campus]['Building'].unique()
    building = st.selectbox("Select Building", buildings)
    
    description = st.text_area("Item Description")
    
    if st.button("Report"):
        if uploaded_file is not None and building and description:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                timestamp = int(time.time())
                image_name = f"{timestamp}_{uploaded_file.name}"
                image_path = os.path.join(image_folder, image_name)
                image.save(image_path)

                # Generate embeddings
                _, image_embedding = encode_text_and_image(image=image)
                image_embedding_str = str(image_embedding.cpu().tolist()) if image_embedding is not None else None

                new_entry = pd.DataFrame({
                    "Image Name": [image_name],
                    "Location": [f"{campus} - {building}"],
                    "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "Embedding": [image_embedding_str],
                    "Description": [description],
                    "Status": ["Lost"]
                })
                
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(excel_file, index=False)

                # Update last added item timestamp
                st.session_state.last_added_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.success(f"Item saved as {os.path.basename(image_path)} with location '{building} ({campus})'")
            except Exception as e:
                st.error(f"Error saving image: {e}")
        else:
            st.warning("Please fill in all fields and upload an image.")
    else:
        st.info("Please fill in all fields and click 'Report' to submit.")

elif option == "Find a Lost Item":
    search_option = st.radio("Search by:", ("Text Description", "Image"))

    text_query = None
    image_query = None

    if search_option == "Text Description":
        text_query = st.text_input("Describe the lost item")
    else:
        uploaded_search_image = st.file_uploader("Upload an image of your lost item", type=["png", "jpg", "jpeg"])
        if uploaded_search_image:
            image_query = Image.open(uploaded_search_image).convert("RGB")
    
    if st.button("Search"):
        if text_query or image_query:
            with st.spinner("Searching for matching items..."):
                best_match, score = find_best_match(text_query=text_query, image_query=image_query)
            
            if best_match and score >= 0.1:
                match_info = df[df['Image Name'] == best_match]
                location = match_info['Location'].values[0] if not match_info.empty else "Unknown"
                st.image(os.path.join(image_folder, best_match), caption=f"Best Match: {best_match} (Similarity Score: {score:.4f})")
                st.success(f"You can collect the item at: {location}")
            else:
                st.warning("No match found! Try refining your description or using another image.")
        else:
            st.warning("Please enter a description or upload an image to search.")

# Add notification for last added item
if 'last_added_time' in st.session_state:
    st.sidebar.markdown("## 🔔 Last Item Added")
    st.sidebar.info(f"Last item was added on: {st.session_state.last_added_time}")

st.sidebar.markdown("Lost & Found Item Search Tool")
