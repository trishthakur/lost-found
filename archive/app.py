import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import time

# Load the CLIP model and processor (only once at startup)
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", padding = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

model, processor, device = load_model()

# Function to encode text and image
def encode_text_and_image(text, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to CLIP expected dimensions
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            text_embedding = outputs.text_embeds
            image_embedding = outputs.image_embeds

        return text_embedding, image_embedding
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None


# Function to find the closest match in a folder
def find_best_match(text_query, image_folder):
    best_match = None
    best_score = -1

    if not os.path.exists(image_folder):
        st.error(f"Image folder '{image_folder}' does not exist.")
        return None, -1

    for img_name in os.listdir(image_folder):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, img_name)
            print(f"Checking image: {image_path}")
            
            text_emb, img_emb = encode_text_and_image(text_query, image_path)
            if text_emb is not None and img_emb is not None:
                score = torch.cosine_similarity(text_emb, img_emb).item()
                
                if score > best_score:
                    best_score = score
                    best_match = img_name

    return best_match, best_score

# Streamlit UI
st.title("Lost and Found Item Matcher")

image_folder = "lost_items"
os.makedirs(image_folder, exist_ok=True)

option = st.radio("Choose an option:", ("Report a Lost Item", "Find a Lost Item"))

if option == "Report a Lost Item":
    uploaded_file = st.file_uploader("Upload an image of the lost item", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            timestamp = int(time.time())
            image_path = os.path.join(image_folder, f"{timestamp}_{uploaded_file.name}")
            image.save(image_path)
            st.success(f"Item saved as {os.path.basename(image_path)}")
        except Exception as e:
            st.error(f"Error saving image: {e}")

elif option == "Find a Lost Item":
    text_query = st.text_input("Describe the lost item")
    if st.button("Search"):
        if text_query:
            with st.spinner("Searching for matching items..."):
                best_match, score = find_best_match(text_query, image_folder)

            if best_match:
                st.image(os.path.join(image_folder, best_match), caption=f"Best Match: {best_match} (Similarity Score: {score:.4f})")
            else:
                st.warning("No match found!")
        else:
            st.warning("Please enter a description to search.")
