import streamlit as st
from rag_pipeline import load_and_index_pdf, create_qa_chain

# Page configuration
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="🏢",
    layout="centered"
)

# Title
st.title("🏢 HR Policy Chatbot")
st.markdown("""
> **AI-powered assistant** that answers your HR policy questions  
> instantly from the official company policy document.
""")

st.divider()

# Load and index PDF only once using session state
if "chain" not in st.session_state:
    with st.spinner("📄 Loading HR Policy document... Please wait..."):
        try:
            vector_store = load_and_index_pdf("data/hr_policy.pdf")
            chain, retriever = create_qa_chain(vector_store)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.success("✅ HR Policy document loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading document: {e}")
            st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask anything about HR policies..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                answer = st.session_state.chain.invoke(prompt)
                st.markdown(answer)

                # Show source pages
                with st.expander("📄 Source Pages Referenced"):
                    docs = st.session_state.retriever.invoke(prompt)
                    for doc in docs:
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Tech Stack:**
    - 🦜 LangChain
    - 🤖 Groq (LLaMA3)
    - 🔍 FAISS Vector Store
    - 🤗 HuggingFace Embeddings
    - 🌐 Streamlit UI
    """)
    st.divider()
    st.markdown("**Sample Questions:**")
    st.markdown("""
    - How many sick leaves do I get?
    - What is the WFH policy?
    - When is salary credited?
    - What is the notice period?
    - How do I raise a grievance?
    """)
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()