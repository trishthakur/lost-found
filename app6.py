import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import time
import pandas as pd
from datetime import datetime
import hashlib
import io
import logging
from mimetypes import guess_type

# SQLAlchemy setup
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gcp_key = st.secrets["GCP_KEY"]

# Save the key to a temporary file
key_path = "service-account.json"
with open(key_path, "w") as f:
    f.write(gcp_key)

# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
# Google Cloud Storage setup
from google.cloud import storage
gcs_bucket_name = "lost_items"  # Replace with your bucket name
storage_client = storage.Client()
bucket = storage_client.bucket(gcs_bucket_name)

# MySQL setup using SQLAlchemy
db_host = "34.59.58.30"  # Replace with your database host (e.g., IP address)
db_user = "user1"  # Replace with your database username
db_password = "root123"  # Replace with your database password
db_name = "hackcu11"

# Database URL
DATABASE_URL = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Define a base class for declarative models
Base = declarative_base()

# Define the LostItems model
class LostItem(Base):
    __tablename__ = "lost_items"

    image_name = Column(String(255), primary_key=True)
    location = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow)
    embedding = Column(Text)
    status = Column(String(255))

# Define the ResolvedItems model
class ResolvedItem(Base):
    __tablename__ = "resolved_items"

    image_name = Column(String(255), primary_key=True)
    location = Column(String(255))
    timestamp = Column(DateTime, default=datetime.utcnow)
    embedding = Column(Text)
    description = Column(Text)
    status = Column(String(255))
    owner_details = Column(Text)

# Define the Users model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, nullable=False)

# Define the Locations model
class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campus = Column(String(255))
    building = Column(String(255))
    contact_email = Column(String(255))

# Create the tables in the database
Base.metadata.create_all(engine)

# Create a Session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Load the CLIP model and processor (only once at startup)
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", padding=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

model, processor, device = load_model()

# Initialize tables
def initialize_tables(db: SessionLocal):
    try:
        # Check if the users table is empty
        if db.query(User).count() == 0:
            admin_username = 'admin'
            admin_password = 'password'  # PLEASE CHANGE THIS!
            hashed_password = hashlib.sha256(admin_password.encode()).hexdigest()
            admin_user = User(username=admin_username, password_hash=hashed_password, is_admin=True)
            db.add(admin_user)
            db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Error initializing tables: {e}")

# Load location data from MySQL
def load_locations(db: SessionLocal):
    try:
        locations_df = pd.read_sql_table("locations", engine)
        locations_df['location'] = locations_df['campus'] + ' - ' + locations_df['building']
        return locations_df
    except Exception as e:
        st.error(f"Error loading locations: {e}")
        return pd.DataFrame()

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(db: SessionLocal, username, password):
    try:
        user = db.query(User).filter(User.username == username).first()
        if user:
            stored_password = user.password_hash
            input_password_hash = hash_password(password)
            return stored_password == input_password_hash
        return False
    except Exception as e:
        st.error(f"Error checking password: {e}")
        return False

def is_admin(db: SessionLocal, username):
   try:
       user = db.query(User).filter(User.username == username).first()
       if user:
           return user.is_admin
       return False
   except Exception as e:
       st.error(f"Error checking admin status: {e}")
       return False

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

def find_best_match(db: SessionLocal, text_query=None, image_query=None):
    best_match = None
    best_score = -1

    try:
        lost_items = db.query(LostItem).all()
        df = pd.DataFrame([(item.image_name, item.embedding) for item in lost_items], columns=['image_name', 'embedding'])
        if 'embedding' not in df.columns:
            st.error("'embedding' column is missing from the table.")
            return None, best_score

        if text_query or image_query:
            query_text_emb, query_img_emb = encode_text_and_image(text=text_query, image=image_query)

            for _, row in df.iterrows():
                if pd.notnull(row['embedding']):
                    try:
                        stored_embedding = torch.tensor(eval(row['embedding']))

                        score = 0
                        if query_text_emb is not None:
                            score += torch.cosine_similarity(query_text_emb, stored_embedding).item()
                        if query_img_emb is not None:
                            score += torch.cosine_similarity(query_img_emb, stored_embedding).item()

                        if score > best_score:
                            best_score = score
                            best_match = row['image_name']
                    except (SyntaxError, NameError, TypeError) as e:
                        st.warning(f"Invalid embedding data for {row['image_name']}. Skipping. Error: {e}")
                        continue

        return best_match, best_score

    except Exception as e:
        st.error(f"Error during search: {e}")
        return None, None

def mark_as_resolved(db: SessionLocal, image_name, owner_details):
    try:
        # Fetch the item from the lost_items table
        lost_item = db.query(LostItem).filter(LostItem.image_name == image_name).first()

        if lost_item:
            # Insert the item into the resolved_items table
            resolved_item = ResolvedItem(
                image_name=lost_item.image_name,
                location=lost_item.location,
                timestamp=lost_item.timestamp,
                embedding=lost_item.embedding,
                description="",
                status='Found',
                owner_details=owner_details
            )
            db.add(resolved_item)

            # Delete the item from the lost_items table
            db.delete(lost_item)

            db.commit()
        else:
            st.warning(f"Item {image_name} not found in lost items.")

    except Exception as e:
        db.rollback()
        st.error(f"Error marking as resolved: {e}")

# Streamlit UI
st.title("Lost and Found Item Matcher")

# Main application logic
option = st.radio("Choose an option:", ("Report a Lost Item", "Find a Lost Item", "Resolve Cases (Staff Only)"))

# Get a database session
with SessionLocal() as db:
    # Initialize tables (call it once)
    initialize_tables(db)

    locations_df = load_locations(db)
    campus_options = locations_df['campus'].unique()

    if option == "Report a Lost Item":
        # Option to upload an image or take a picture
        image_option = st.radio("Choose an image source:", ("Upload from computer", "Take a picture"))

        uploaded_file = None
        image = None

        if image_option == "Upload from computer":
            uploaded_file = st.file_uploader("Upload an image of the lost item", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                original_filename = uploaded_file.name
            else:
                image = None
                original_filename = None
        else:
            captured_image = st.camera_input("Take a picture of the lost item")
            if captured_image:
                image = Image.open(captured_image).convert("RGB")
                original_filename = "captured_image.jpg"
            else:
                image = None
                original_filename = None

        campus = st.selectbox("Select Campus", campus_options)
        buildings = locations_df[locations_df['campus'] == campus]['building'].unique()
        building = st.selectbox("Select Building", buildings)

        if st.button("Report"):
            if image is not None and building and original_filename:
                try:
                    timestamp = int(time.time())
                    image_name = f"{timestamp}_{os.path.splitext(original_filename)[0]}.jpg" #force .jpg
                    logger.info(f"Generated image name: {image_name}")

                    # Convert image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()

                    # Upload image to GCS
                    blob = bucket.blob(image_name)
                    blob.upload_from_string(img_byte_arr, content_type='image/jpeg')
                    logger.info(f"Uploaded image to GCS with content type: image/jpeg")

                    # Generate embeddings
                    _, image_embedding = encode_text_and_image(image=image)
                    image_embedding_str = str(image_embedding.cpu().tolist()) if image_embedding is not None else None
                    location = f"{campus} - {building}"

                    # Insert item into MySQL using SQLAlchemy
                    lost_item = LostItem(
                        image_name=image_name,
                        location=location,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        embedding=image_embedding_str,
                        status="Lost"
                    )
                    db.add(lost_item)
                    db.commit()
                    st.success(f"Item saved as {image_name} in GCS and recorded in the database with location '{building} ({campus})'")

                except Exception as e:
                    db.rollback()
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
                    best_match, score = find_best_match(db, text_query=text_query, image_query=image_query)

                if best_match and score >= 0.1:
                    # Construct the public URL for the image
                    public_url = f"https://storage.googleapis.com/{gcs_bucket_name}/{best_match}"

                    # Retrieve location and email information from the database
                    lost_item = db.query(LostItem).filter(LostItem.image_name == best_match).first()
                    if lost_item:
                        location = lost_item.location
                    else:
                        location = "Unknown"

                    email = locations_df[locations_df['location'] == location]['contact_email'].values[0] if location in locations_df['location'].values else "Email not found"

                    # Display the image using st.image
                    st.image(public_url, caption=f"Best Match: {best_match} (Similarity Score: {score:.4f})")
                    st.success(f"Found a potential match!  Image: {best_match}, Location: {location}, Contact Email: {email}")
                else:
                    st.info("No good matches found.")

    elif option == "Resolve Cases (Staff Only)":
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False

        if not st.session_state.logged_in:
            st.subheader("Staff Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.button("Login")

            if login_button:
                if check_password(db, username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Logged in successfully")
                    st.rerun()
                else:
                    st.error("Incorrect username or password")

        if st.session_state.logged_in:
            st.subheader(f"Welcome, {st.session_state.username}")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.rerun()

            st.subheader("Resolve Cases")
            try:
                lost_items = db.query(LostItem).filter(LostItem.status == 'Lost').all()
                if not lost_items:
                    st.info("No lost items to resolve.")
                else:
                    for item in lost_items:
                        with st.expander(f"Item: {item.image_name}"):
                            st.write(f"Location: {item.location}")
                            st.write(f"Timestamp: {item.timestamp}")

                            # Construct the public URL for the image
                            public_url = f"https://storage.googleapis.com/{gcs_bucket_name}/{item.image_name}"
                            st.image(public_url, width=200)

                            owner_details = st.text_input("Enter owner details:", key=f"owner_{item.image_name}")
                            if st.button(f"Resolve", key=f"resolve_{item.image_name}"):
                                if owner_details:
                                    mark_as_resolved(db, item.image_name, owner_details)
                                    st.success(f"Item {item.image_name} marked as resolved")
                                    st.rerun()
                                else:
                                    st.warning("Please enter owner details before resolving.")

            except Exception as e:
                st.error(f"Error retrieving lost items: {e}")

    # Display recently reported item
    try:
        # Query the most recent lost item
        last_reported_item = db.query(LostItem).order_by(LostItem.timestamp.desc()).first()

        if last_reported_item:
            st.sidebar.markdown("## 🔔 Last Reported Item")
            st.sidebar.info(f"Last item reported at {last_reported_item.timestamp}")
        else:
            st.sidebar.info("No items have been reported yet.")
    except Exception as e:
        st.error(f"Error retrieving last reported item: {e}")

