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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Cloud Storage setup
from google.cloud import storage
gcs_bucket_name = "lost_items"  # Replace with your bucket name
storage_client = storage.Client()
bucket = storage_client.bucket(gcs_bucket_name)

# MySQL setup with SQLAlchemy
from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String, Boolean, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import mysql.connector

db_host = "34.59.58.30"  # Replace with your database host (e.g., IP address)
db_user = "user1"  # Replace with your database username
db_password = "root123"  # Replace with your database password
db_name = "hackcu11"

# SQLAlchemy setup
DATABASE_URL = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define the SQLAlchemy models
class LostItem(Base):
    __tablename__ = "lost_items"

    image_name = Column(String(255), primary_key=True)
    location = Column(String(255))
    timestamp = Column(String(255))
    embedding = Column(Text)
    status = Column(String(255))


class ResolvedItem(Base):
    __tablename__ = "resolved_items"

    image_name = Column(String(255), primary_key=True)
    location = Column(String(255))
    timestamp = Column(String(255))
    embedding = Column(Text)
    description = Column(Text)
    status = Column(String(255))
    owner_details = Column(Text)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, nullable=False)


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campus = Column(String(255))
    building = Column(String(255))
    contact_email = Column(String(255))


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Load the CLIP model and processor (only once at startup)
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", padding=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device


model, processor, device = load_model()

# Table names (using class names now)
lost_items_table = LostItem.__tablename__
resolved_items_table = ResolvedItem.__tablename__
locations_table = Location.__tablename__
users_table = User.__tablename__


# Initialize tables (using SQLAlchemy)
def initialize_tables():
    db = SessionLocal()
    try:
        # Check if the users table is empty
        user_count = db.query(User).count()
        if user_count == 0:
            admin_username = 'admin'
            admin_password = 'password'  # PLEASE CHANGE THIS!
            hashed_password = hashlib.sha256(admin_password.encode()).hexdigest()

            # Create and add the admin user
            admin_user = User(username=admin_username, password_hash=hashed_password, is_admin=True)
            db.add(admin_user)
            db.commit()
    except Exception as e:
        st.error(f"Error initializing tables: {e}")
        db.rollback()
    finally:
        db.close()


# Load location data from MySQL using SQLAlchemy
def load_locations():
    db = SessionLocal()
    try:
        locations = db.query(Location).all()
        locations_data = []
        for loc in locations:
            locations_data.append(
                {
                    "campus": loc.campus,
                    "building": loc.building,
                    "contact_email": loc.contact_email,
                    "location": f"{loc.campus} - {loc.building}",
                    "id": loc.id,
                }
            )
        locations_df = pd.DataFrame(locations_data)
        return locations_df
    except Exception as e:
        st.error(f"Error loading locations: {e}")
        return pd.DataFrame()
    finally:
        db.close()


locations_df = load_locations()

# Create campus options
campus_options = locations_df['campus'].unique()


# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(username, password):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user:
            input_password_hash = hash_password(password)
            return user.password_hash == input_password_hash
        return False
    except Exception as e:
        st.error(f"Error checking password: {e}")
        return False
    finally:
        db.close()


def is_admin(username):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user:
            return user.is_admin
        return False
    except Exception as e:
        st.error(f"Error checking admin status: {e}")
        return False
    finally:
        db.close()


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


def find_best_match(text_query=None, image_query=None):
    best_match = None
    best_score = -1

    db = SessionLocal()
    try:
        # Fetch embeddings using SQLAlchemy
        lost_items = db.query(LostItem.image_name, LostItem.embedding).all()
        df = pd.DataFrame(lost_items, columns=['image_name', 'embedding'])
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
        return None, best_score
    finally:
        db.close()


def mark_as_resolved(image_name, owner_details):
    db = SessionLocal()
    try:
        # Fetch the item from the lost_items table using SQLAlchemy
        item = db.query(LostItem).filter(LostItem.image_name == image_name).first()

        if item:
            # Create a ResolvedItem object from the LostItem object
            resolved_item = ResolvedItem(
                image_name=item.image_name,
                location=item.location,
                timestamp=item.timestamp,
                embedding=item.embedding,
                description="",
                status="Found",
                owner_details=owner_details
            )

            # Add the resolved item to the resolved_items table
            db.add(resolved_item)

            # Delete the item from the lost_items table
            db.delete(item)

            db.commit()
        else:
            st.warning(f"Item {image_name} not found in lost items.")

    except Exception as e:
        db.rollback()
        st.error(f"Error marking as resolved: {e}")
    finally:
        db.close()


# Initialize tables (call it once)
initialize_tables()

# Streamlit UI
st.title("Lost and Found Item Matcher")

# Main application logic
option = st.radio("Choose an option:", ("Report a Lost Item", "Find a Lost Item", "Resolve Cases (Staff Only)"))

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
                image_name = f"{timestamp}_{os.path.splitext(original_filename)[0]}.jpg"  # force .jpg
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
                db = SessionLocal()
                try:
                    lost_item = LostItem(
                        image_name=image_name,
                        location=location,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        embedding=image_embedding_str,
                        status="Lost"
                    )
                    db.add(lost_item)
                    db.commit()
                    st.success(
                        f"Item saved as {image_name} in GCS and recorded in the database with location '{building} ({campus})'")

                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving item to database: {e}")
                finally:
                    db.close()
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
                # Construct the public URL for the image
                public_url = f"https://storage.googleapis.com/{gcs_bucket_name}/{best_match}"

                # Retrieve location and email information from the database
                db = SessionLocal()
                try:
                    # Use SQLAlchemy to fetch the location
                    item = db.query(LostItem).filter(LostItem.image_name == best_match).first()

                    if item:
                        location = item.location
                    else:
                        location = "Unknown"

                    # Retrieve contact email from locations_df
                    email = locations_df[locations_df['location'] == location]['contact_email'].values[0] if location in locations_df[
                        'location'].values else "Email not found"

                    # Display the image using st.image
                    st.image(public_url, caption=f"Best Match: {best_match} (Similarity Score: {score:.4f})")
                    st.success(f"You can collect the item at: {location}")
                    st.info(f"Contact Email: {email}")

                except Exception as e:
                    st.error(f"Error retrieving item information: {e}")
                finally:
                    db.close()
            else:
                st.info("No matching items found.")

elif option == "Resolve Cases (Staff Only)":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if check_password(username, password) and is_admin(username):
            st.success("Login successful!")
            # List unresolved items
            db = SessionLocal()
            try:
                unresolved_items = db.query(LostItem).all()
                if unresolved_items:
                    st.write("### Unresolved Items")
                    for item in unresolved_items:
                        # Construct the public URL for the image
                        public_url = f"https://storage.googleapis.com/{gcs_bucket_name}/{item.image_name}"
                        st.image(public_url, caption=item.image_name, width=200)  # Display image
                        st.write(f"Location: {item.location}")
                        st.write(f"Timestamp: {item.timestamp}")
                        owner_details = st.text_input(f"Owner Details for {item.image_name}")
                        resolve_button = st.button(f"Mark {item.image_name} as Resolved", key=f"resolve_{item.image_name}")

                        if resolve_button:
                            mark_as_resolved(item.image_name, owner_details)
                            st.success(f"{item.image_name} marked as resolved.")
                            st.experimental_rerun()  # Refresh the page to update the list

                else:
                    st.info("No unresolved items.")
            except Exception as e:
                st.error(f"Error retrieving unresolved items: {e}")
            finally:
                db.close()
        else:
            st.error("Invalid username or password. Staff only.")
