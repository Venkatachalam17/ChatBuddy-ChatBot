import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="ChatBuddy Smart Tutor", 
    page_icon="🧠", 
    layout="centered"
)

st.title("🧠 ChatBuddy: Smart Technical Tutor")
st.write("Optimized for typo-correction, abbreviation handling, and structured point-by-point breakdowns.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared! Ask me a question or test a technical abbreviation."}
        ]
        st.rerun()

# 1. LOAD MODEL WITH CACHING
@st.cache_resource
def load_text_model():
    return pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct")

with st.spinner("Loading AI model into memory..."):
    text_chatbot = load_text_model()

# 2. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your smart tech tutor. I automatically handle typos and abbreviations, and structure my answers with points, examples, and flowcharts!"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Chat Input & Structured Prompt Generation
if user_input := st.chat_input("Ask a question (typos & abbreviations allowed!)..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing query, fixing typos, and structuring response..."):
            try:
                # 🧠 ADVANCED SYSTEM PROMPT FOR TYPOS, ABBREVIATIONS & STRUCTURE
                system_instruction = (
                    "You are an expert, highly intelligent technical AI tutor. "
                    "Follow these rules strictly for every response:\n"
                    "1. **Query Normalization**: If the user's input contains typos or abbreviations, briefly clarify what term or abbreviation you are addressing.\n"
                    "2. **Overview**: Provide a short, precise definition.\n"
                    "3. **Key Points**: Break down the core concepts using bullet points.\n"
                    "4. **Example**: Provide a concrete code snippet or real-world example.\n"
                    "5. **Flow / Mindmap**: Provide a text-based hierarchical tree or flow representation using markdown tree lines."
                )
                
                prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
                
                # Keep history clean and focused (last 4 messages)
                for msg in st.session_state.messages[-4:]:
                    if msg["role"] == "user":
                        prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                    else:
                        prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
                
                prompt += "<|im_start|>assistant\n"

                outputs = text_chatbot(
                    prompt,
                    max_new_tokens=300,    # Room for bullet points, code, and mindmap
                    do_sample=True,
                    temperature=0.25,      # Low temperature for strict factual accuracy & structure
                    top_p=0.85,
                    repetition_penalty=1.2
                )
                
                full_text = outputs[0]["generated_text"]
                bot_reply = full_text[len(prompt):].split("<|im_end|>")[0].strip()
                
                if not bot_reply:
                    bot_reply = "I parsed your query, but let me re-organize that for you with clear points and an example..."

            except Exception as e:
                bot_reply = f"⚠️ Error: {str(e)}"

            st.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})