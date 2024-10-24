import streamlit as st
from pymongo import MongoClient
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_DB_CONN_URL")

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client['Users']  # Replace with your database name
user_collection = db['Credentials']  # Collection for storing user accounts

# Create password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Function to hash password
def hash_password(password):
    return pwd_context.hash(password)

# Function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Function to check if the username exists
def check_username_exists(username):
    user = user_collection.find_one({"username": username})
    return user is not None

# Function to add a new user
def add_user(username, password):
    hashed_password = hash_password(password)
    user_collection.insert_one({"username": username, "password": hashed_password})

# Page selection
page = st.sidebar.selectbox("Select Page", options=["Login", "Create Account"])

if page == "Create Account":
    st.header("Create Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if st.button("Create Account"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif check_username_exists(username):
            st.error("Username already exists.")
        else:
            add_user(username, password)
            st.success("Account created successfully! You can now log in.")

elif page == "Login":
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if not check_username_exists(username):
            st.error("Username does not exist.")
        else:
            user = user_collection.find_one({"username": username})
            if verify_password(password, user['password']):
                st.success("Login successful!")
                # Redirect to the diabetes prediction page or store the session state
                st.session_state.username = username  # Store username in session state
                st.write(f"Welcome, {st.session_state.username}!")
                # You can add diabetes prediction functionality here
            else:
                st.error("Incorrect password.")