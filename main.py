import streamlit as st
import requests
import json
from typing import List, Dict
import time

# Set page config
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        align-self: flex-start;
        margin-right: 20%;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

class HuggingFaceChat:
    def __init__(self, api_token: str, model_name: str = "microsoft/DialoGPT-medium"):
        self.api_token = api_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    def query(self, payload: Dict) -> Dict:
        """Send request to Hugging Face API"""
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return {"error": str(e)}
    
    def chat(self, message: str, conversation_history: List[Dict] = None) -> str:
        """Generate chat response"""
        if conversation_history is None:
            conversation_history = []
        
        # Prepare the input text with conversation history
        input_text = ""
        for turn in conversation_history[-5:]:  # Keep last 5 turns for context
            input_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        input_text += f"User: {message}\nAssistant:"
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": 50256
            }
        }
        
        result = self.query(payload)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
            # Extract only the new assistant response
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
                return response if response else "I'm sorry, I couldn't generate a proper response."
            else:
                return "I'm sorry, I couldn't generate a proper response."
        else:
            return "I'm sorry, I couldn't generate a response."

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "chat_instance" not in st.session_state:
        st.session_state.chat_instance = None

def display_chat_messages():
    """Display chat messages"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">You</div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">🤖 Assistant</div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

def main():
    st.title("🤖 AI Chat Assistant")
    st.markdown("Chat with your Hugging Face AI assistant!")
    
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get API token from secrets or user input
        api_token = st.secrets.get("HUGGINGFACE_API_TOKEN", "")
        
        if not api_token:
            api_token = st.text_input(
                "Hugging Face API Token",
                type="password",
                help="Enter your Hugging Face API token"
            )
        else:
            st.success("✅ API Token loaded from secrets")
        
        # Model selection
        model_options = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill",
            "facebook/blenderbot-1B-distill",
            "microsoft/DialoGPT-small"
        ]
        
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            help="Choose the Hugging Face model for chat"
        )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
        
        # Chat statistics
        if st.session_state.messages:
            st.markdown("---")
            st.markdown("📊 **Chat Statistics**")
            st.metric("Total Messages", len(st.session_state.messages))
            st.metric("Your Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
            st.metric("Assistant Messages", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
    
    # Main chat interface
    if not api_token:
        st.warning("⚠️ Please provide your Hugging Face API token to start chatting.")
        st.info("You can get your API token from: https://huggingface.co/settings/tokens")
        return
    
    # Initialize chat instance
    if st.session_state.chat_instance is None or st.session_state.chat_instance.model_name != selected_model:
        st.session_state.chat_instance = HuggingFaceChat(api_token, selected_model)
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        display_chat_messages()
    
    # Chat input at the bottom
    st.markdown("---")
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...",
            key="user_input",
            placeholder="Ask me anything!",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    # Handle user input
    if (send_button or user_input) and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Show typing indicator
        with st.spinner("🤖 Assistant is typing..."):
            # Generate response
            response = st.session_state.chat_instance.chat(
                user_input,
                st.session_state.conversation_history
            )
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Update conversation history
        st.session_state.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        # Clear input and rerun
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [Streamlit](https://streamlit.io) and [Hugging Face](https://huggingface.co)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
