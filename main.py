import streamlit as st
import requests
import json
from typing import List, Dict
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# Set page config
st.set_page_config(
    page_title="Notion Notes Converter",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI and Notion-like styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #f8f9fa;
        border-left: 4px solid #007acc;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #28a745;
    }
    .message-header {
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #2c3e50;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .notion-output {
        background-color: #fafbfc;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Segoe UI', system-ui, sans-serif;
        line-height: 1.6;
    }
    .notion-output h1, .notion-output h2, .notion-output h3 {
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .notion-output h1 { font-size: 1.8rem; font-weight: 700; }
    .notion-output h2 { font-size: 1.4rem; font-weight: 600; }
    .notion-output h3 { font-size: 1.2rem; font-weight: 500; }
    .notion-output ul, .notion-output ol {
        margin: 1rem 0;
        padding-left: 1.5rem;
    }
    .notion-output li {
        margin-bottom: 0.3rem;
    }
    .notion-output blockquote {
        border-left: 3px solid #007acc;
        margin: 1rem 0;
        padding-left: 1rem;
        color: #555;
        font-style: italic;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 2px dashed #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class NotionNotesConverter:
    def _init_(self, api_token: str, model_name: str = "gpt2"):
        self.api_token = api_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        
        # System prompt for Notion-style conversion
        self.system_prompt = """You are a professional note-taking assistant that converts messy text, transcripts, or raw ideas into clean, structured Notion-style notes. Your task is to:

1. *Structure the content* with proper headings (# ## ###)
2. *Create bullet points* for key information
3. *Add summaries* at the beginning or end
4. *Organize information* logically
5. *Use formatting* like bold, italic, and blockquotes appropriately
6. *Extract action items* if present
7. *Create tables* when data is structured

Always respond in clean Markdown format that would look great in Notion. Make the notes scannable and well-organized."""
    
    def query(self, payload: Dict) -> Dict:
        """Send request to Hugging Face API with proper error handling"""
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload,
                timeout=30  # Add timeout
            )
            
            # Check if the response is successful
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                return {"error": "Model is currently loading. Please try again in a few moments."}
            elif response.status_code == 401:
                return {"error": "Invalid API token. Please check your Hugging Face API token."}
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded. Please wait before making another request."}
            else:
                return {"error": f"API request failed with status {response.status_code}: {response.text}"}
                
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Please try again."}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error. Please check your internet connection."}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def convert_to_notes(self, messy_text: str) -> str:
        """Convert messy text to structured Notion-style notes"""
        
        # Validate input
        if not messy_text or len(messy_text.strip()) < 10:
            return "❌ *Error:* Please provide more text content (at least 10 characters)."
        
        # Truncate if too long to avoid API limits
        if len(messy_text) > 1000:
            messy_text = messy_text[:1000] + "..."
            st.warning("⚠ Text was truncated to 1000 characters to avoid API limits.")
        
        # Create a focused prompt for note conversion
        conversion_prompt = f"""Convert the following messy text into clean, structured Notion-style notes with proper headings, bullet points, and formatting:

Input: {messy_text}

Output: """
        
        payload = {
            "inputs": conversion_prompt,
            "parameters": {
                "max_new_tokens": 300,  # Reduced to avoid timeout
                "temperature": 0.3,
                "top_p": 0.8,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "return_full_text": False  # Only return generated text
            }
        }
        
        # Add loading indicator
        with st.spinner("🤖 AI is processing your text..."):
            result = self.query(payload)
        
        # Handle API errors
        if "error" in result:
            error_msg = result["error"]
            st.error(f"❌ *API Error:* {error_msg}")
            
            # Provide fallback for certain errors
            if "loading" in error_msg.lower():
                st.info("💡 *Tip:* Try switching to a different model or wait a few minutes.")
            
            return self.create_fallback_notes(messy_text)
        
        # Process successful response
        try:
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                generated_text = result["generated_text"]
            else:
                generated_text = str(result)
            
            # Clean up the generated text
            if "Output:" in generated_text:
                response = generated_text.split("Output:")[-1].strip()
            else:
                response = generated_text.strip()
            
            # Remove any remaining prompt text
            if "Input:" in response:
                response = response.split("Input:")[-1].strip()
            
            # Validate response quality
            if len(response) < 20 or not any(char in response for char in ['#', '*', '-']):
                st.warning("⚠ AI response seems incomplete. Using structured fallback.")
                return self.create_fallback_notes(messy_text)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing API response: {e}")
            return self.create_fallback_notes(messy_text)
    
    def create_fallback_notes(self, text: str) -> str:
        """Create a basic structured note when AI fails"""
        try:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            notes = "# 📝 Structured Notes\n\n"
            
            # Add summary
            notes += "## 📋 Summary\n"
            notes += f"- Content contains {len(sentences)} key points\n"
            notes += f"- Organized into {min(len(lines), 10)} main items\n"
            notes += f"- Word count: approximately {len(text.split())} words\n\n"
            
            # Add main content
            notes += "## 📌 Key Points\n\n"
            
            if lines:
                for i, line in enumerate(lines[:8], 1):  # Limit to 8 lines
                    if len(line) > 10:  # Only include substantial lines
                        notes += f"{i}.** {line}\n\n"
            else:
                # Fallback to sentences
                for i, sentence in enumerate(sentences[:6], 1):  # Limit to 6 sentences
                    if len(sentence) > 15:
                        notes += f"- {sentence.strip()}.\n"
            
            # Add action items if keywords found
            action_keywords = ['need', 'should', 'must', 'todo', 'action', 'follow up', 'complete']
            if any(keyword in text.lower() for keyword in action_keywords):
                notes += "\n## ✅ Action Items\n"
                notes += "- Review and organize the above points\n"
                notes += "- Follow up on any mentioned tasks\n"
                notes += "- Add more details where needed\n"
            
            return notes
            
        except Exception as e:
            logger.error(f"Error in fallback notes creation: {e}")
            return f"# 📝 Notes\n\n## Content\n\n{text}\n\n*Note: Automatic structuring failed, showing original content.*"

def initialize_session_state():
    """Initialize session state variables"""
    if "conversion_history" not in st.session_state:
        st.session_state.conversion_history = []
    if "converter_instance" not in st.session_state:
        st.session_state.converter_instance = None

def display_conversion_history():
    """Display conversion history with error handling"""
    try:
        if st.session_state.conversion_history:
            st.subheader("📚 Recent Conversions")
            
            # Show only last 3 conversions to avoid UI clutter
            recent_conversions = list(reversed(st.session_state.conversion_history[-3:]))
            
            for i, conversion in enumerate(recent_conversions):
                timestamp = conversion.get('timestamp', 'Unknown time')
                truncated_input = conversion['input'][:50] + "..." if len(conversion['input']) > 50 else conversion['input']
                
                with st.expander(f"🕒 {timestamp} - {truncated_input}", expanded=False):
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("📝 Original Text:")
                        st.text_area(
                            "", 
                            value=conversion['input'], 
                            height=120, 
                            disabled=True, 
                            key=f"input_history_{i}_{timestamp}"
                        )
                    
                    with col2:
                        st.markdown("✨ Structured Notes:")
                        st.markdown(conversion["output"])
    except Exception as e:
        logger.error(f"Error displaying history: {e}")
        st.error("Error loading conversion history.")

def validate_api_token(token: str) -> bool:
    """Validate API token format"""
    if not token:
        return False
    # Basic validation - HF tokens typically start with 'hf_'
    return len(token) > 20 and (token.startswith('hf_') or len(token) > 30)

def main():
    try:
        st.title("📝 Notion Notes Converter")
        st.markdown("*Transform messy text, transcripts, or raw ideas into clean, structured Notion-style notes!*")
        
        initialize_session_state()
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("⚙ Configuration")
            
            # Get API token
            api_token = None
            
            # Try to get from secrets first
            try:
                if hasattr(st, 'secrets') and 'HUGGINGFACE_API_TOKEN' in st.secrets:
                    api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
                    st.success("✅ API Token loaded from secrets")
            except Exception:
                pass
            
            # If no token from secrets, ask user
            if not api_token:
                api_token = st.text_input(
                    "🔑 Hugging Face API Token",
                    type="password",
                    help="Get your free API token from: https://huggingface.co/settings/tokens",
                    placeholder="hf_..."
                )
                
                if api_token and not validate_api_token(api_token):
                    st.warning("⚠ API token format looks incorrect. Make sure you copied the full token.")
            
            # Model selection - Focus on reliable free models
            model_options = [
                "gpt2",
                "distilgpt2",
                "gpt2-medium",
                "microsoft/DialoGPT-small",
                "facebook/opt-350m",
                "EleutherAI/gpt-neo-125M"
            ]
            
            selected_model = st.selectbox(
                "🤖 Select Model",
                model_options,
                help="Smaller models are faster and more reliable"
            )
            
            # Model info
            model_info = {
                "gpt2": "Fast, reliable, good for basic structuring",
                "distilgpt2": "Fastest option, good for simple notes",
                "gpt2-medium": "Better quality, slower",
                "microsoft/DialoGPT-small": "Good for conversational text",
                "facebook/opt-350m": "Alternative option",
                "EleutherAI/gpt-neo-125M": "Lightweight alternative"
            }
            
            if selected_model in model_info:
                st.info(f"ℹ {model_info[selected_model]}")
            
            # Features info
            st.markdown("---")
            st.markdown("📋 *Features*")
            st.success("""
            ✅ Auto-structured headings  
            ✅ Bullet point organization  
            ✅ Clean formatting  
            ✅ Action item extraction  
            ✅ Notion-ready output
            """)
            
            # Clear history
            if st.button("🗑 Clear History", help="Clear all conversion history"):
                st.session_state.conversion_history = []
                st.rerun()
            
            # Statistics
            if st.session_state.conversion_history:
                st.markdown("---")
                st.markdown("📊 *Stats*")
                st.metric("Total Conversions", len(st.session_state.conversion_history))
        
        # Main interface
        if not api_token or not validate_api_token(api_token):
            st.warning("⚠ Please provide a valid Hugging Face API token to start.")
            
            st.info("""
            *How to get your API token:*
            1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
            2. Create a free account if needed
            3. Create a new token with 'Read' permissions
            4. Copy and paste it above
            """)
            
            st.markdown("---")
            st.markdown("### 🎯 What this app does:")
            st.markdown("""
            - 📝 *Converts messy text* into clean, structured notes
            - 🎯 *Creates proper headings* and bullet points  
            - 📋 *Generates summaries* for long content
            - ✅ *Extracts action items* automatically
            - 🎨 *Formats for Notion* with markdown styling
            - 📊 *Organizes information* logically
            """)
            return
        
        # Initialize converter
        if (st.session_state.converter_instance is None or 
            st.session_state.converter_instance.model_name != selected_model):
            st.session_state.converter_instance = NotionNotesConverter(api_token, selected_model)
        
        # Input section
        st.markdown("### 📝 Input Your Text")
        
        # Example texts
        example_texts = {
            "Custom Text": "",
            "Meeting Notes": "talked about project timeline john said we need to finish by friday sarah mentioned budget concerns mike will handle the design phase need to follow up with client about requirements also discussed marketing strategy for q4",
            "Lecture Notes": "today we covered machine learning basics supervised learning uses labeled data unsupervised learning finds patterns reinforcement learning learns through rewards neural networks mimic brain structure deep learning uses multiple layers",
            "Research Ideas": "renewable energy solutions solar panel efficiency wind power scalability battery storage advances nuclear fusion potential carbon capture government policies adoption rates cost analysis market trends"
        }
        
        selected_example = st.selectbox("Choose example or use custom:", list(example_texts.keys()))
        
        user_input = st.text_area(
            "Enter your text:",
            value=example_texts[selected_example],
            height=150,
            placeholder="Paste your messy text, meeting notes, or ideas here...",
            max_chars=2000  # Limit input length
        )
        
        # Character count
        if user_input:
            char_count = len(user_input)
            st.caption(f"Characters: {char_count}/2000")
            if char_count > 1500:
                st.warning("⚠ Long text may be truncated for API limits.")
        
        # Convert button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            convert_button = st.button(
                "🚀 Convert to Notion Notes", 
                use_container_width=True, 
                type="primary",
                disabled=not user_input or len(user_input.strip()) < 10
            )
        
        # Handle conversion
        if convert_button and user_input.strip():
            try:
                # Generate structured notes
                structured_notes = st.session_state.converter_instance.convert_to_notes(user_input)
                
                # Display result
                st.markdown("---")
                st.subheader("✅ Your Structured Notes")
                
                # Display with styling
                st.markdown(structured_notes)
                
                # Copy section
                st.markdown("---")
                st.markdown("📋 Copy to Notion:")
                st.code(structured_notes, language="markdown")
                st.info("💡 Copy the markdown above and paste directly into Notion!")
                
                # Save to history
                st.session_state.conversion_history.append({
                    "input": user_input,
                    "output": structured_notes,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M"),
                    "model": selected_model
                })
                
                st.success("🎉 Conversion completed!")
                
            except Exception as e:
                logger.error(f"Conversion error: {e}")
                st.error(f"❌ Conversion failed: {str(e)}")
                st.info("💡 Try using a different model or shorter text.")
        
        # Display history
        if st.session_state.conversion_history:
            st.markdown("---")
            display_conversion_history()
        
        # Footer
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            *💡 Tips:*
            - Keep text under 1000 characters
            - Include context and details
            - Try different models for variety
            - Copy output directly to Notion
            """)
        
        with col2:
            st.markdown("""
            *🔧 Troubleshooting:*
            - Check API token if errors occur
            - Try smaller models for speed
            - Wait if model is loading
            - Use fallback if AI fails
            """)
    
    except Exception as e:
        logger.error(f"Main app error: {e}")
        st.error("❌ Application error occurred. Please refresh and try again.")
        st.exception(e)

if _name_ == "_main_":
    main()
