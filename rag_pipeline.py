import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

def load_and_index_pdf(pdf_path):
    """Load PDF, split into chunks, create FAISS vector store"""

    # Step 1: Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Step 3: Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Step 4: Store in FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def create_qa_chain(vector_store):
    """Create the RAG question-answering chain using LCEL"""

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Custom prompt template
    prompt = PromptTemplate.from_template("""
    You are a helpful HR assistant. Use the following context from the
    HR policy document to answer the employee's question accurately.
    If the answer is not in the context, say "I couldn't find that
    information in the HR policy document."

    Context: {context}

    Question: {question}

    Answer:""")

    # Helper to format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever