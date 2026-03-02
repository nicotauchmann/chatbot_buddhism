import os
from pathlib import Path

import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# ---------- Streamlit config ----------
st.set_page_config(page_title="Buddhism Info Bot", page_icon="🧘", layout="centered")

TOP_K = 3


# ---------- Helpers ----------
def get_secret(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


def to_llama_history(messages: list[dict]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in messages:
        role = MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
        out.append(ChatMessage(role=role, content=m["content"]))
    return out


BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vector_index"  # commit this folder to GitHub
EMBED_CACHE_DIR = (
    BASE_DIR / ".cache" / "embeddings"
)  # ephemeral cache on Streamlit Cloud


# ---------- Resources (cached across reruns) ----------
@st.cache_resource
def make_llm() -> Groq:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY (set it in Streamlit Cloud Secrets).")
    return Groq(model="llama-3.3-70b-versatile", api_key=api_key)


@st.cache_resource
def make_embeddings() -> HuggingFaceEmbedding:
    embedding_model = get_secret(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbedding(
        model_name=embedding_model, cache_folder=str(EMBED_CACHE_DIR)
    )


@st.cache_resource
def load_vector_index():
    if not VECTOR_DIR.exists():
        raise RuntimeError(
            f"Missing '{VECTOR_DIR.name}' folder in the repo. "
            "Commit your persisted LlamaIndex storage (vector_index/) to GitHub."
        )
    embeddings = make_embeddings()
    storage_context = StorageContext.from_defaults(persist_dir=str(VECTOR_DIR))
    return load_index_from_storage(storage_context, embed_model=embeddings)


def get_chat_engine():
    # Per-user (per-session) engine + memory
    if "chat_engine" not in st.session_state:
        llm = make_llm()
        index = load_vector_index()

        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        # This is the “system/context” instruction the condense_plus_context engine uses
        context_prompt = (
            "You are a helpful assistant who answers questions about Buddhism.\n"
            "You must speak like young professor (calm, humble, thoughtful).\n\n"
            "Here are the relevant documents for the context:\n"
            "{context_str}\n\n"
            "Instruction: Use the previous chat history or the context above to answer.\n"
            "If the context does not contain the answer, say you cannot find it in the documents."
        )

        # IMPORTANT: do NOT pass retriever=... here, or you can hit the “multiple values” error.
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            llm=llm,
            memory=memory,
            similarity_top_k=TOP_K,
            context_prompt=context_prompt,
            verbose=False,
        )
    return st.session_state.chat_engine


# ---------- UI ----------
st.title("Buddhism Info Bot")
st.write(
    "Ask a question about Buddhism. The bot answers only from buddhistic literature."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()
        st.rerun()

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input (auto-clears after submit)
user_msg = st.chat_input("What is your question?")

if user_msg:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    try:
        chat_engine = get_chat_engine()

        # “Reset memory each question and pass only this user’s history”
        chat_engine.reset()
        llama_history = to_llama_history(
            st.session_state.messages[:-1]
        )  # history BEFORE this user message

        streaming = chat_engine.stream_chat(user_msg, chat_history=llama_history)

        chunks: list[str] = []
        with st.chat_message("assistant"):

            def gen():
                for tok in streaming.response_gen:
                    chunks.append(tok)
                    yield tok

            st.write_stream(gen())

        assistant_text = "".join(chunks).strip()
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text}
        )

    except Exception as e:
        st.error(str(e))
