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

# MySQL setup
import mysql.connector
db_host = "34.59.58.30"  # Replace with your database host (e.g., IP address)
db_user = "user1"  # Replace with your database username
db_password = "root123"  # Replace with your database password
db_name = "hackcu11"

def get_db_connection():
    try:
        connection = mysql.connector.connect(host=db_host,
                                             user=db_user,
                                             password=db_password,
                                             database=db_name)
        return connection
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        return None

# Load the CLIP model and processor (only once at startup)
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", padding=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

model, processor, device = load_model()

# Table names
lost_items_table = "lost_items"
resolved_items_table = "resolved_items"
locations_table = "locations"
users_table = "users"

# Initialize tables
def initialize_tables():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{lost_items_table}` (
                    `image_name` VARCHAR(255) PRIMARY KEY,
                    `location` VARCHAR(255),
                    `timestamp` VARCHAR(255),
                    `embedding` TEXT,
                    `status` VARCHAR(255)
                )
            """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{resolved_items_table}` (
                    `image_name` VARCHAR(255) PRIMARY KEY,
                    `location` VARCHAR(255),
                    `timestamp` VARCHAR(255),
                    `embedding` TEXT,
                    `description` TEXT,
                    `status` VARCHAR(255),
                    `owner_details` TEXT
                )
            """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{users_table}` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `username` VARCHAR(255) UNIQUE NOT NULL,
                    `password_hash` VARCHAR(255) NOT NULL,
                    `is_admin` BOOLEAN NOT NULL
                )
            """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{locations_table}` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `campus` VARCHAR(255),
                    `building` VARCHAR(255),
                    `contact_email` VARCHAR(255)
                )
            """)

            # Insert initial admin user if the table is empty
            cursor.execute(f"SELECT COUNT(*) FROM `{users_table}`")
            if cursor.fetchone()[0] == 0:
                admin_username = 'admin'
                admin_password = 'password'  # PLEASE CHANGE THIS!
                hashed_password = hashlib.sha256(admin_password.encode()).hexdigest()
                cursor.execute(f"""
                    INSERT INTO `{users_table}` (`username`, `password_hash`, `is_admin`)
                    VALUES (%s, %s, %s)
                """, (admin_username, hashed_password, True))

            conn.commit()
        except mysql.connector.Error as e:
            st.error(f"Error creating/initializing tables: {e}")
        finally:
            cursor.close()
            conn.close()

# Load location data from MySQL
def load_locations():
    conn = get_db_connection()
    if conn:
        try:
            locations_df = pd.read_sql(f"SELECT * FROM `{locations_table}`", conn)
            locations_df['location'] = locations_df['campus'] + ' - ' + locations_df['building']
            return locations_df
        except mysql.connector.Error as e:
            st.error(f"Error loading locations: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    else:
        return pd.DataFrame()

locations_df = load_locations()

# Create campus options
campus_options = locations_df['campus'].unique()

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(username, password):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            query = f"SELECT password_hash FROM `{users_table}` WHERE username = %s"
            cursor.execute(query, (username,))
            result = cursor.fetchone()
            if result:
                stored_password = result[0]
                input_password_hash = hash_password(password)
                return stored_password == input_password_hash
            return False
        except mysql.connector.Error as e:
            st.error(f"Error checking password: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    else:
        return False

def is_admin(username):
   conn = get_db_connection()
   if conn:
       cursor = conn.cursor()
       try:
           query = f"SELECT is_admin FROM `{users_table}` WHERE username = %s"
           cursor.execute(query, (username,))
           result = cursor.fetchone()
           if result:
               return bool(result[0])
           return False
       except mysql.connector.Error as e:
           st.error(f"Error checking admin status: {e}")
           return False
       finally:
           cursor.close()
           conn.close()
   else:
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

def find_best_match(text_query=None, image_query=None):
    best_match = None
    best_score = -1

    conn = get_db_connection()
    if not conn:
        return None, best_score

    try:
        df = pd.read_sql(f"SELECT `image_name`, `embedding` FROM `{lost_items_table}`", conn)

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

    except mysql.connector.Error as e:
        st.error(f"Error during search: {e}")
        return None, best_score
    finally:
        conn.close()

def mark_as_resolved(image_name, owner_details):
    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor()
    try:
        # Fetch the item from the lost_items table
        query = f"SELECT * FROM `{lost_items_table}` WHERE `image_name` = %s"
        cursor.execute(query, (image_name,))
        item = cursor.fetchone()

        if item:
            # Insert the item into the resolved_items table
            insert_query = f"""
                INSERT INTO `{resolved_items_table}` 
                (`image_name`, `location`, `timestamp`, `embedding`, `description`, `status`, `owner_details`)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (item[0], item[1], item[2], item[3], "", 'Found', owner_details))

            # Delete the item from the lost_items table
            delete_query = f"DELETE FROM `{lost_items_table}` WHERE `image_name` = %s"
            cursor.execute(delete_query, (image_name,))

            conn.commit()
        else:
            st.warning(f"Item {image_name} not found in lost items.")

    except mysql.connector.Error as e:
        conn.rollback()
        st.error(f"Error marking as resolved: {e}")
    finally:
        cursor.close()
        conn.close()

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

                # Insert item into MySQL
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    try:
                        query = f"""
                            INSERT INTO `{lost_items_table}` 
                            (`image_name`, `location`, `timestamp`, `embedding`, `status`)
                            VALUES (%s, %s, %s, %s, %s)
                        """
                        cursor.execute(query, (image_name, location, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_embedding_str, "Lost"))
                        conn.commit()
                        st.success(f"Item saved as {image_name} in GCS and recorded in the database with location '{building} ({campus})'")

                    except mysql.connector.Error as e:
                        conn.rollback()
                        st.error(f"Error saving item to database: {e}")
                    finally:
                        cursor.close()
                        conn.close()
                else:
                    st.error("Failed to connect to the database.")

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
                conn = get_db_connection()
                if conn:
                    try:
                        query = f"SELECT `location` FROM `{lost_items_table}` WHERE `image_name` = %s"
                        cursor = conn.cursor()
                        cursor.execute(query, (best_match,))
                        result = cursor.fetchone()

                        if result:
                            location = result[0]
                        else:
                            location = "Unknown"

                        email = locations_df[locations_df['location'] == location]['contact_email'].values[0] if location in locations_df['location'].values else "Email not found"

                        # Display the image using st.image
                        st.image(public_url, caption=f"Best Match: {best_match} (Similarity Score: {score:.4f})")
                        st.success(f"You can collect the item at: {location}")
                        st.info(f"Contact Email: {email}")

                    except mysql.connector.Error as e:
                        st.error(f"Error retrieving item information: {e}")
                    finally:
                        conn.close()
                else:
                    st.error("Failed to connect to the database.")
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
            if check_password(username, password):
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
        conn = get_db_connection()
        if conn:
            try:
                query = f"SELECT * FROM `{lost_items_table}` WHERE `status` = 'Lost'"
                lost_items = pd.read_sql(query, conn)
                
                if lost_items.empty:
                    st.info("No lost items to resolve.")
                else:
                    for _, item in lost_items.iterrows():
                        with st.expander(f"Item: {item['image_name']}"):
                            st.write(f"Location: {item['location']}")
                            st.write(f"Timestamp: {item['timestamp']}")

                            # Construct the public URL for the image
                            public_url = f"https://storage.googleapis.com/{gcs_bucket_name}/{item['image_name']}"
                            st.image(public_url, width=200)
                            
                            owner_details = st.text_input("Enter owner details:", key=f"owner_{item['image_name']}")
                            if st.button(f"Resolve", key=f"resolve_{item['image_name']}"):
                                if owner_details:
                                    mark_as_resolved(item['image_name'], owner_details)
                                    st.success(f"Item {item['image_name']} marked as resolved")
                                    st.rerun()
                                else:
                                    st.warning("Please enter owner details before resolving.")

            except mysql.connector.Error as e:
                st.error(f"Error retrieving lost items: {e}")
            finally:
                conn.close()
        else:
            st.error("Failed to connect to the database.")

# Display recently reported item
conn = get_db_connection()
if conn:
    try:
        query = f"SELECT * FROM `{lost_items_table}` ORDER BY `timestamp` DESC LIMIT 1"
        last_reported_item = pd.read_sql(query, conn)
        
        if not last_reported_item.empty:
            st.sidebar.markdown("## 🔔 Last Reported Item")
            st.sidebar.info(f"Last item reported at {last_reported_item['timestamp'].values[0]}")
        else:
            st.sidebar.info("No items have been reported yet.")
    except mysql.connector.Error as e:
        st.error(f"Error retrieving last reported item: {e}")
    finally:
        conn.close()
else:
    st.sidebar.info("Failed to connect to the database.")
