import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
import hashlib

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
excel_file = "lost_items.csv"
resolved_file = "resolved_items.csv"
locations_file = "locations.xlsx"
users_file = "users.csv"

# Initialize CSV files if not present
if not Path(excel_file).exists():
    pd.DataFrame(columns=["Image Name", "Location", "Timestamp", "Embedding", "Description", "Status", "Owner Details"]).to_csv(excel_file, index=False)

if not Path(resolved_file).exists():
    pd.DataFrame(columns=["Image Name", "Location", "Timestamp", "Embedding", "Description", "Status", "Owner Details"]).to_csv(resolved_file, index=False)

if not Path(users_file).exists():
    pd.DataFrame(columns=["Username", "Password", "Is_Admin"]).to_csv(users_file, index=False)

# Load location data
locations_df = pd.read_excel(locations_file)
# Create a "Location" column in locations_df
locations_df['Location'] = locations_df['Campus'] + ' - ' + locations_df['Building']
campus_options = locations_df['Campus'].unique()

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(username, password):
    users_df = pd.read_csv(users_file)
    print(f"Checking password for username: {username}")
    print(f"Users in database: {users_df['Username'].tolist()}")
    user = users_df[users_df['Username'] == username]
    if not user.empty:
        stored_password = user.iloc[0]['Password']
        input_password_hash = hash_password(password)
        print(f"Stored password hash: {stored_password}")
        print(f"Input password hash: {input_password_hash}")
        return stored_password == input_password_hash
    return False

def is_admin(username):
    users_df = pd.read_csv(users_file)
    user = users_df[users_df['Username'] == username]
    if not user.empty:
        return user.iloc[0]['Is_Admin']
    return False

def print_users_file():
    if os.path.exists(users_file):
        users_df = pd.read_csv(users_file)
        print("Contents of users file:")
        print(users_df)
    else:
        print("Users file does not exist")

print_users_file()

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

def mark_as_resolved(image_name, owner_details):
    df = pd.read_csv(excel_file)
    resolved_df = pd.read_csv(resolved_file)
    
    item = df[df['Image Name'] == image_name].iloc[0]
    item['Status'] = 'Found'
    item['Owner Details'] = owner_details
    
    resolved_df = pd.concat([resolved_df, pd.DataFrame([item])], ignore_index=True)
    resolved_df.to_csv(resolved_file, index=False)
    
    df = df[df['Image Name'] != image_name]
    df.to_csv(excel_file, index=False)

# Streamlit UI
st.title("Lost and Found Item Matcher")

# Main application logic
option = st.radio("Choose an option:", ("Report a Lost Item", "Find a Lost Item", "Resolve Cases (Staff Only)"))

if option == "Report a Lost Item":
    # Option to upload an image or take a picture
    image_option = st.radio("Choose an image source:", ("Upload from computer", "Take a picture"))

    uploaded_file = None  # Define uploaded_file outside the if block
    image = None  # Define image outside the if block

    if image_option == "Upload from computer":
        uploaded_file = st.file_uploader("Upload an image of the lost item", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        else:
            image = None
    else:
        captured_image = st.camera_input("Take a picture of the lost item")
        if captured_image:
            image = Image.open(captured_image).convert("RGB")
        else:
            image = None
    
    campus = st.selectbox("Select Campus", campus_options)
    buildings = locations_df[locations_df['Campus'] == campus]['Building'].unique()
    building = st.selectbox("Select Building", buildings)
    
    if st.button("Report"):
        if image is not None and building:
            try:
                timestamp = int(time.time())
                image_name = f"{timestamp}_{'captured_image.jpg' if image_option == 'Take a picture' else uploaded_file.name if uploaded_file else 'default.jpg'}" # Ensure filename is available
                image_path = os.path.join(image_folder, image_name)
                image.save(image_path)

                # Generate embeddings
                _, image_embedding = encode_text_and_image(image=image)
                image_embedding_str = str(image_embedding.cpu().tolist()) if image_embedding is not None else None
                location = f"{campus} - {building}"

                new_entry = pd.DataFrame({
                    "Image Name": [image_name],
                    "Location": [location],
                    "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "Embedding": [image_embedding_str],
                    "Description": [""],
                    "Status": ["Lost"],
                    "Owner Details": [""]
                })
                
                df = pd.read_csv(excel_file)
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(excel_file, index=False)

                st.success(f"Item saved as {os.path.basename(image_path)} with location '{building} ({campus})'")
            except Exception as e:
                st.error(f"Error saving image: {e}")
        else:
            st.warning("Please fill in all fields and upload or take a picture.")

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
                df = pd.read_csv(excel_file)
                match_info = df[df['Image Name'] == best_match]
                location = match_info['Location'].values[0] if not match_info.empty else "Unknown"
                # Get email from locations_df
                email = locations_df[locations_df['Location'] == location]['Contact Email'].values[0] if location in locations_df['Location'].values else "Email not found"
                st.image(os.path.join(image_folder, best_match), caption=f"Best Match: {best_match} (Similarity Score: {score:.4f})")
                st.success(f"You can collect the item at: {location}")
                st.info(f"Contact Email: {email}")

            else:
                st.warning("No match found! Try refining your description or using another image.")
        else:
            st.warning("Please enter a description or upload an image to search.")

elif option == "Resolve Cases (Staff Only)":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.subheader("Staff Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            print(f"Login attempt with username: {username}")
            if check_password(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully")
                st.rerun()
            else:
                st.error("Incorrect username or password")
                print("Login failed")

    if st.session_state.logged_in:
        st.subheader(f"Welcome, {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        
        st.subheader("Resolve Cases")
        df = pd.read_csv(excel_file)
        lost_items = df[df['Status'] == 'Lost']
        
        if lost_items.empty:
            st.info("No lost items to resolve.")
        else:
            for _, item in lost_items.iterrows():
                with st.expander(f"Item: {item['Image Name']}"):
                    st.write(f"Location: {item['Location']}")
                    st.write(f"Timestamp: {item['Timestamp']}")
                    st.image(os.path.join(image_folder, item['Image Name']), width=200)
                    
                    owner_details = st.text_input("Enter owner details:", key=f"owner_{item['Image Name']}")
                    if st.button(f"Resolve", key=f"resolve_{item['Image Name']}"):
                        if owner_details:
                            mark_as_resolved(item['Image Name'], owner_details)
                            st.success(f"Item {item['Image Name']} marked as resolved")
                            st.rerun()
                        else:
                            st.warning("Please enter owner details before resolving.")

# Display recently reported item
if Path(excel_file).exists():
    df = pd.read_csv(excel_file)
    last_reported_item = df.sort_values('Timestamp', ascending=False).head(1)
    if not last_reported_item.empty:
        st.sidebar.markdown("## 🔔 Last Reported Item")
        st.sidebar.info(f"Last item reported at {last_reported_item['Timestamp'].values[0]}")
    else:
        st.sidebar.info("No items have been reported yet.")
