from dotenv import load_dotenv
import os
import json
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
st.set_page_config(page_title="EDUCATION ASSISTANT",page_icon="🎓")
st.title("🎓 EDUCATION ASSISTANT")


with st.sidebar:
    st.subheader("⚙️ Controls")
    model_name = st.selectbox(
        "ChatGroq Model",
        ["llama-3.1-8b-instant","groq/compound","openai/gpt-oss-120b"],
        index=1
    )
    temperature = st.slider("temperature (creativity)", 0.1,0.9,0.4)
    max_tokens = st.slider("max_tokens",100,400,350)

    system_prompt = st.text_area(
        "System Prompt Rules",
        value = "You are a friendly and knowledgeable education assistant for university students. "
              "Answer their questions clearly and supportively using simple English."
    )

    if st.button("🧹Clear Chat"):
        st.session_state.pop("history",None)
        st.rerun()

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Add it to your .env or deployment secrets.")
    st.stop()

# initilize single history
if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()

# LLM + prompt + chain
# chat groq reads GROQ_API_KEY from .env

llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens
)

# Role - structured prompt: System -> History -> Human

prompt = ChatPromptTemplate.from_messages([
    ("system","{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])

chain = prompt | llm | StrOutputParser()

Chat_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: st.session_state.history,
    input_messages_key = "input",
    history_messages_key="history"
)

for msg in st.session_state.history.messages:
    role = getattr(msg,"type",None) or getattr(msg,"role","")
    content = msg.content
    if role == "human":
        st.chat_message( "user").write(content)
    elif role in ("ai","assistant"):
        st.chat_message("assistant").write(content)
    
user_input = st.chat_input("Type your message....")

if user_input:
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            response_text = Chat_with_history.invoke(
                {"input": user_input,"system_prompt":system_prompt},
                config={"configurable":{"session_id":"default"}}
            )
        except Exception as e:
            st.error(f"Model error: {e}")
            response_text = ""

        typed = "" 
        for ch in response_text:
            typed += ch
            placeholder.markdown(typed)
if st.session_state.history.messages:
    export = []
    for m in st.session_state.history.messages:
        role = getattr(m,"type",None) or getattr (m,"role","")
        if role == "human":
            export.append({"role":"user","text":m.content})
        elif role in("ai","assistant"):
            export.append({"role":"assistant","text":m.content})
    st.download_button(
        "Download chat JSON",
        data = json.dumps(export,ensure_ascii=False,indent=2),
        file_name="chat_history.json",
        mime= "application/json",
        use_container_width=True
    )
