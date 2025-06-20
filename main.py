import streamlit as st
import requests
import json
from typing import List, Dict
import time

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
    def __init__(self, api_token: str, model_name: str = "gpt2"):
        self.api_token = api_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        
        # System prompt for Notion-style conversion
        self.system_prompt = """You are a professional note-taking assistant that converts messy text, transcripts, or raw ideas into clean, structured Notion-style notes. Your task is to:

1. **Structure the content** with proper headings (# ## ###)
2. **Create bullet points** for key information
3. **Add summaries** at the beginning or end
4. **Organize information** logically
5. **Use formatting** like bold, italic, and blockquotes appropriately
6. **Extract action items** if present
7. **Create tables** when data is structured

Always respond in clean Markdown format that would look great in Notion. Make the notes scannable and well-organized."""
    
    def query(self, payload: Dict) -> Dict:
        """Send request to Hugging Face API"""
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return {"error": str(e)}
    
    def convert_to_notes(self, messy_text: str) -> str:
        """Convert messy text to structured Notion-style notes"""
        
        # Create a focused prompt for note conversion
        conversion_prompt = f"""Convert the following messy text into clean, structured Notion-style notes with proper headings, bullet points, and formatting:

Input: {messy_text}

Output:
        
        payload = {
            "inputs": conversion_prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.3,  # Lower temperature for more structured output
                "top_p": 0.8,
                "do_sample": True,
                "pad_token_id": 50256,
                "repetition_penalty": 1.1
            }
        }
        
        result = self.query(payload)
        
        if "error" in result:
            return f"❌ **Error:** {result['error']}\n\nPlease try again with a different model or check your API token."
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
            
            # Extract the response part
            if "Output:" in generated_text:
                response = generated_text.split("Output:")[-1].strip()
            else:
                response = generated_text.strip()
            
            # If response is empty or too short, provide a structured fallback
            if len(response) < 50:
                return self.create_fallback_notes(messy_text)
            
            return response
        else:
            return self.create_fallback_notes(messy_text)
    
    def create_fallback_notes(self, text: str) -> str:
        """Create a basic structured note when AI fails"""
        lines = text.split('\n')
        notes = "# 📝 Structured Notes\n\n"
        notes += "## Summary\n"
        notes += f"- Raw content provided for structuring\n"
        notes += f"- Contains {len(lines)} lines of text\n\n"
        notes += "## Content\n\n"
        
        for i, line in enumerate(lines[:10], 1):  # Limit to first 10 lines
            if line.strip():
                notes += f"- **Point {i}:** {line.strip()}\n"
        
        if len(lines) > 10:
            notes += f"\n*... and {len(lines) - 10} more lines*\n"
        
        return notes

def initialize_session_state():
    """Initialize session state variables"""
    if "conversion_history" not in st.session_state:
        st.session_state.conversion_history = []
    if "converter_instance" not in st.session_state:
        st.session_state.converter_instance = None

def display_conversion_history():
    """Display conversion history"""
    if st.session_state.conversion_history:
        st.subheader("📚 Conversion History")
        
        for i, conversion in enumerate(reversed(st.session_state.conversion_history[-5:])):  # Show last 5
            with st.expander(f"Conversion {len(st.session_state.conversion_history) - i}: {conversion['timestamp']}", expanded=False):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Text:**")
                    st.text_area("", value=conversion['input'][:200] + "..." if len(conversion['input']) > 200 else conversion['input'], 
                               height=100, disabled=True, key=f"input_{i}")
                
                with col2:
                    st.markdown("**Structured Notes:**")
                    st.markdown(f'<div class="notion-output">{conversion["output"]}</div>', unsafe_allow_html=True)

def main():
    st.title("📝 Notion Notes Converter")
    st.markdown("**Transform messy text, transcripts, or raw ideas into clean, structured Notion-style notes!**")
    
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get API token from secrets or user input
        try:
            api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
            st.success("✅ API Token loaded from secrets")
        except (KeyError, AttributeError):
            api_token = st.text_input(
                "Hugging Face API Token",
                type="password",
                help="Enter your Hugging Face API token"
            )
        
        # Model selection - FREE models that work with API tokens
        model_options = [
            "gpt2",
            "gpt2-medium", 
            "gpt2-large",
            "distilgpt2",
            "facebook/opt-350m",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "bigscience/bloom-560m",
            "t5-small",
            "t5-base"
        ]
        
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            help="Choose the Hugging Face model for note conversion"
        )
        
        # Conversion settings
        st.markdown("---")
        st.markdown("📋 **Conversion Features**")
        st.info("""
        - Auto-structured headings
        - Bullet point organization  
        - Summary generation
        - Action item extraction
        - Clean Markdown formatting
        - Notion-ready output
        """)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.conversion_history = []
            st.rerun()
        
        # Conversion statistics
        if st.session_state.conversion_history:
            st.markdown("---")
            st.markdown("📊 **Statistics**")
            st.metric("Total Conversions", len(st.session_state.conversion_history))
    
    # Main conversion interface
    if not api_token:
        st.warning("⚠️ Please provide your Hugging Face API token to start converting notes.")
        st.info("Get your API token from: https://huggingface.co/settings/tokens")
        st.markdown("---")
        st.markdown("### 🎯 What this app does:")
        st.markdown("""
        - **Converts messy text** into clean, structured notes
        - **Creates proper headings** and bullet points
        - **Generates summaries** for long content
        - **Extracts action items** automatically
        - **Formats for Notion** with markdown styling
        - **Organizes information** logically
        """)
        return
    
    # Initialize converter instance
    if st.session_state.converter_instance is None or st.session_state.converter_instance.model_name != selected_model:
        st.session_state.converter_instance = NotionNotesConverter(api_token, selected_model)
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("📝 Input Your Messy Text")
    
    # Example texts for quick testing
    example_texts = {
        "Meeting Notes": "talked about project timeline john said we need to finish by friday sarah mentioned budget concerns mike will handle the design phase need to follow up with client about requirements also discussed marketing strategy for q4",
        "Lecture Transcript": "so today we're going to talk about machine learning which is basically when computers learn patterns from data without being explicitly programmed there are three main types supervised learning unsupervised learning and reinforcement learning supervised learning uses labeled data",
        "Research Ideas": "looking into sustainable energy solutions solar panels efficiency has improved wind power scalability issues battery storage technology advances nuclear fusion potential carbon capture methods need more research government policies impact on adoption",
        "Custom Text": ""
    }
    
    selected_example = st.selectbox("Choose an example or use custom text:", list(example_texts.keys()))
    
    if selected_example == "Custom Text":
        user_input = st.text_area(
            "Paste your messy text, transcript, or raw ideas here:",
            height=200,
            placeholder="Paste your unstructured text here... It can be meeting notes, lecture transcripts, random thoughts, or any messy content you want to organize into clean Notion-style notes."
        )
    else:
        user_input = st.text_area(
            "Edit this example or replace with your own text:",
            value=example_texts[selected_example],
            height=200
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Convert button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        convert_button = st.button("🚀 Convert to Notion Notes", use_container_width=True, type="primary")
    
    # Handle conversion
    if convert_button and user_input.strip():
        if len(user_input.strip()) < 10:
            st.warning("⚠️ Please provide more text to convert (at least 10 characters).")
            return
            
        # Show conversion progress
        with st.spinner("✨ Converting your text into structured Notion-style notes..."):
            # Generate structured notes
            structured_notes = st.session_state.converter_instance.convert_to_notes(user_input)
        
        # Display result
        st.markdown("---")
        st.subheader("✅ Your Structured Notion Notes")
        
        # Display the converted notes with Notion styling
        st.markdown(f'<div class="notion-output">{structured_notes}</div>', unsafe_allow_html=True)
        
        # Copy to clipboard button
        st.code(structured_notes, language="markdown")
        st.info("💡 **Tip:** Copy the markdown above and paste it directly into Notion!")
        
        # Save to history
        st.session_state.conversion_history.append({
            "input": user_input,
            "output": structured_notes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": selected_model
        })
        
        st.success("🎉 Conversion completed! Your notes are ready for Notion.")
    
    # Display conversion history
    if st.session_state.conversion_history:
        st.markdown("---")
        display_conversion_history()
    
    # Footer with tips
    st.markdown("---")
    st.markdown("### Tips for Best Results:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Input Tips:**
        - Longer text works better
        - Include context and details
        - Do not worry about formatting
        - Mix different topics is OK
        """)
    
    with col2:
        st.markdown("""
        **Output Features:**
        - Clean headings and structure
        - Bullet points for key info
        - Summaries and action items
        - Ready for Notion import
        """)
    
    st.markdown("---")
    st.markdown(
        "Built with love using Streamlit and Hugging Face | Perfect for Notion users!",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
