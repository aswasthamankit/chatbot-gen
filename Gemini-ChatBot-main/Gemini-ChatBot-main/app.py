
import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors


# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Chat with Google Gemini",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 2. LOAD ENVIRONMENT VARIABLES
# ==========================================

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2")
]

API_KEYS = [key.strip() for key in API_KEYS if key and key.strip()]

if not API_KEYS:
    st.error("No Gemini API keys found. Check your .env file.")
    st.stop()


# ==========================================
# 3. GEMINI CLIENT INITIALIZATION
# ==========================================

@st.cache_resource
def get_gemini_clients(keys):
    return [
        genai.Client(api_key=key)
        for key in keys
    ]


clients = get_gemini_clients(tuple(API_KEYS))


# ==========================================
# 4. MODEL CONFIGURATION
# ==========================================

MODEL_NAME = "gemini-3.8-flash"


# ==========================================
# 5. SESSION STATE INITIALIZATION
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = clients[0].chats.create(
        model=MODEL_NAME
    )

if "active_client" not in st.session_state:
    st.session_state.active_client = 0

if "request_count" not in st.session_state:
    st.session_state.request_count = 0


# ==========================================
# 6. APPLICATION UI
# ==========================================

st.title("🤖 Chat with Google Gemini")

st.caption("Powered by Google Gemini AI")

with st.sidebar:

    st.header("Chat Settings")

    st.caption(f"Model: {MODEL_NAME}")
    st.caption(f"Available API keys: {len(clients)}")
    st.caption("Streaming: Enabled")
    st.caption("Automatic retries: Enabled")

    if st.button("Clear Chat History"):

        st.session_state.messages = []

        st.session_state.chat = clients[
            st.session_state.active_client
        ].chats.create(model=MODEL_NAME)

        st.rerun()


# ==========================================
# 7. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# 8. GEMINI RESPONSE WITH RETRIES
# ==========================================

def generate_response(prompt, max_retries=4):

    # Start with the currently active client
    start_client = st.session_state.active_client

    last_error = None

    for attempt in range(max_retries):

        # Rotate through available API keys on retry
        client_index = (start_client + attempt) % len(clients)

        try:

            # Create a new chat using the selected client
            chat = clients[client_index].chats.create(
                model=MODEL_NAME,
                history=[
                    {
                        "role": msg["role"] if msg["role"] == "user" else "model",
                        "parts": [{"text": msg["content"]}]
                    }
                    for msg in st.session_state.messages[:-1]
                ]
            )

            # Generate streaming response
            response = chat.send_message_stream(prompt)

            full_response = ""

            for chunk in response:

                if chunk.text:

                    full_response += chunk.text
                    yield chunk.text

            # Update active client after successful response
            st.session_state.active_client = client_index

            return

        except errors.ServerError as e:

            last_error = e

            if "503" not in str(e) and "UNAVAILABLE" not in str(e):
                raise

            if attempt == max_retries - 1:
                break

            wait_time = min(2 ** (attempt + 1), 16)

            st.warning(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

        except errors.ClientError:
            # Authentication and invalid request errors
            raise

    if last_error:
        raise last_error


# ==========================================
# 9. USER CHAT INPUT
# ==========================================

if prompt := st.chat_input(
    "I possess a well of knowledge. What would you like to know?"
):

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):

        try:

            response = st.write_stream(
                generate_response(prompt)
            )

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except errors.APIError as e:

            st.error(f"Gemini API Error: {e}")

        except Exception as e:

            st.error(f"Unexpected Error: {e}")
