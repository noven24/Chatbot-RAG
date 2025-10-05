# Import the necessary libraries
import streamlit as st  # For creating the web app interface
from google import genai  # For interacting with the Google Gemini API

# --- 1. Page Configuration and Title ---

# Set the title and a caption for the web page
st.title("💬 Gemini Chatbot")
st.caption("A simple and friendly chat using Google's Gemini Flash model")

# --- 2. Sidebar for Settings ---

# The sidebar is now simpler, only containing the reset button.
with st.sidebar:
    st.subheader("Controls")
    # Create a button to reset the conversation.
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

# --- 3. API Key and Client Initialization (MODIFIED SECTION) ---

# Get the API key from Streamlit's secrets management.
# This is the secure and recommended way to handle API keys.
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    # If the key is not found in secrets, show an error and stop.
    st.error("Google AI API Key not found. Please add it to your Streamlit secrets.", icon="🗝️")
    st.stop()

# Initialize the Gemini client.
# This block now only runs once per session since the key doesn't change.
if "genai_client" not in st.session_state:
    try:
        # Configure and create the Gemini client.
        st.session_state.genai_client = genai.Client(api_key=google_api_key)
    except Exception as e:
        # If the key is invalid or another error occurs, show an error and stop.
        st.error(f"Failed to initialize Gemini client: {e}")
        st.stop()

# --- 4. Chat History Management ---

# Initialize the chat session if it doesn't already exist in memory.
if "chat" not in st.session_state:
    # Create a new chat instance using the 'gemini-1.5-flash' model.
    st.session_state.chat = st.session_state.genai_client.chats.create(model="gemini-2.5-flash")

# Initialize the message history (as a list) if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle the reset button click.
if reset_button:
    # If the reset button is clicked, clear the chat object and message history from memory.
    st.session_state.pop("chat", None)
    st.session_state.pop("messages", None)
    # st.rerun() tells Streamlit to refresh the page from the top.
    st.rerun()

# --- 5. Display Past Messages ---

# Loop through every message currently stored in the session state.
for msg in st.session_state.messages:
    # For each message, create a chat message bubble with the appropriate role ("user" or "assistant").
    with st.chat_message(msg["role"]):
        # Display the content of the message using Markdown for nice formatting.
        st.markdown(msg["content"])

# --- 6. Handle User Input and API Communication ---

# Create a chat input box at the bottom of the page.
# The user's typed message will be stored in the 'prompt' variable.
prompt = st.chat_input("Type your message here...")

if prompt:
    # 1. Add the user's message to our message history list.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 2. Display the user's message on the screen immediately for a responsive feel.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Get the assistant's response.
    # Use a 'try...except' block to gracefully handle potential errors.
    try:
        # Send the user's prompt to the Gemini API.
        response = st.session_state.chat.send_message(prompt)
        
        # Safely get the text from the response object.
        if hasattr(response, "text"):
            answer = response.text
        else:
            answer = str(response)

    except Exception as e:
        # If any error occurs, create an error message to display to the user.
        answer = f"An error occurred: {e}"
        st.error(answer) # Show error in the app UI as well

    # 4. Display the assistant's response.
    with st.chat_message("assistant"):
        st.markdown(answer)
    # 5. Add the assistant's response to the message history list.
    st.session_state.messages.append({"role": "assistant", "content": answer})
