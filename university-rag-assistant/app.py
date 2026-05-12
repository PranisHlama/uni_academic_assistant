import json
import faiss
import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="University Chat Assistant")
st.title("Pranish Uni Assistant")

# Load JSON
@st.cache_data
def load_uni_data():
    with open("uni_data.json", "r", encoding="utf-8") as file:
        return json.load(file)

uni_data = load_uni_data()

# Flatten JSON
@st.cache_data
def extract_all_qa(data):
    rows = []

    def extract_qa(obj, section="general"):
        if isinstance(obj, dict):
            if "question" in obj and "answer" in obj:
                rows.append({
                    "section": section,
                    "question": obj["question"],
                    "answer": obj["answer"]
                })

            for key, value in obj.items():
                extract_qa(value, key)

        elif isinstance(obj, list):
            for item in obj:
                extract_qa(item, section)

    extract_qa(data)
    return rows

rows = extract_all_qa(uni_data)

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Build FAISS index once
@st.cache_resource
def build_index(rows):
    questions = [row["question"] for row in rows]

    embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings

index, question_embeddings = build_index(rows)

# RAG answer
def rag_answer(user_query):
    user_embedding = model.encode(
        [user_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(user_embedding, 2)

    best_score = float(scores[0][0])

    if best_score < 0.45:
        return "I don't have enough information in university data.", []

    contexts = []
    for i in indices[0]:
        item = rows[i]
        contexts.append(
            f"Section: {item['section']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a university support assistant.

Rules:
- Answer using only the context.
- Keep the answer short and direct.
- If the answer is not in the context, say:
  "I don't have enough information in university data."

Context:
{context_text}

Question:
{user_query}
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": "You are a concise university assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 80,
            "num_ctx": 1024
        }
    )

    return response["message"]["content"], contexts

# Session State
if "chat" not in st.session_state:
    st.session_state.chat = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "current_chat_index" not in st.session_state:
    st.session_state.current_chat_index = None

# Sidebar Chat History
with st.sidebar:
    st.title("Chat History")

    if st.button("+ New Chat"):
        if st.session_state.chat:
            title = st.session_state.chat[0]["content"][:30]
            if len(st.session_state.chat[0]["content"]) > 30:
                title += "..."

            st.session_state.chat_sessions.append({
                "title": title,
                "messages": st.session_state.chat.copy()
            })

        st.session_state.chat = []
        st.session_state.current_chat_index = None
        st.rerun()

    st.divider()

    if st.session_state.chat_sessions:
        for idx, session in enumerate(reversed(st.session_state.chat_sessions)):
            original_index = len(st.session_state.chat_sessions) - 1 - idx

            if st.button(session["title"], key=f"chat_{original_index}"):
                st.session_state.chat = session["messages"].copy()
                st.session_state.current_chat_index = original_index
                st.rerun()
    else:
        st.caption("No chats yet.")

    st.divider()

    if st.button("Clear History"):
        st.session_state.chat = []
        st.session_state.chat_sessions = []
        st.session_state.current_chat_index = None
        st.rerun()

# Display Chat Messages
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Ask your question here")

if user_input:
    st.session_state.chat.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = rag_answer(user_input)

        st.markdown(answer)

        if sources:
            with st.expander("Sources used"):
                for source in sources:
                    st.info(source)

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer
    })

    # If this is an old chat, update it after new message
    if st.session_state.current_chat_index is not None:
        st.session_state.chat_sessions[st.session_state.current_chat_index]["messages"] = st.session_state.chat.copy()

    st.rerun()