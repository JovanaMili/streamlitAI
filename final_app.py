import streamlit as st
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModel
from pathlib import Path
import tempfile
from datetime import datetime
import torch

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from utils_img import get_base64_of_local_image

def get_chroma_client():
    # Use new ChromaDB PersistentClient API (DuckDB/Parquet, local storage)
    return chromadb.PersistentClient(path=".chromadb")

# --- FIX: Reset collection when problems occur

def reset_database():
    client = get_chroma_client()
    try:
        client.delete_collection("docs")
    except:
        pass
    return client.create_collection("docs")

client = get_chroma_client()
collection = client.get_or_create_collection("docs")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    return embedding.tolist()

def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")

def show_document_manager():
    """Display document manager interface"""
    st.subheader("📋 Manage Documents")
    
    if not st.session_state.converted_docs:
        st.info("No documents uploaded yet.")
        return
    
    # Show each document with delete button
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        
        with col2:
            # Preview button
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        
        with col3:
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.converted_docs.pop(i)
                # Rebuild ChromaDB collection from current docs
                collection.delete()
                for doc in st.session_state.converted_docs:
                    embedding = embed_text(doc['content'])
                    collection.add(
                        documents=[doc['content']],
                        embeddings=[embedding],
                        ids=[doc['filename']]
                    )
                st.rerun()
        
        # Show preview if requested
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()

def add_to_search_history(question, answer, source):
    """Add search to history"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Add new search to beginning of list
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    
    # Keep only last 10 searches
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    """Display search history"""
    st.subheader("🕒 Recent Searches")
    
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("No searches yet.")
        return
    
    for i, search in enumerate(st.session_state.search_history):
        with st.expander(f"Q: {search['question'][:50]}... ({search['timestamp']})"):
            st.write("**Question:**", search['question'])
            st.write("**Answer:**", search['answer'])
            st.write("**Source:**", search['source'])

# --- Custom CSS for professional look ---
def add_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@700&display=swap');
    html, body {
        height: 100% !important;
        min-height: 100vh !important;
        background: #B7D9B1 !important; /* RAL 6019 pastel green */
        box-sizing: border-box !important;
    }
    [data-testid="stAppViewContainer"], .stApp, .block-container {
        background: transparent !important;
        box-shadow: none !important;
    }
    .main-header {
        font-family: 'Baloo 2', cursive, sans-serif;
        font-size: 2.7rem;
        color: #3A6B35;
        text-align: center;
        margin-bottom: 0;
        margin-top: 0.5rem;
        padding: 1.2rem;
        background: rgba(255,255,255,0.97);
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(58,107,53,0.10);
        letter-spacing: 1px;
        border: 2.5px solid #3A6B35;
        display: block;
    }
    .main-header-spacer {
        height: 3.5rem;
        width: 100%;
        display: block;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #B7D9B1 0%, #3A6B35 100%);
        color: #222;
        border: 2px solid #3A6B35;
        box-shadow: 0 2px 8px rgba(58,107,53,0.10);
        transition: background 0.3s;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #3A6B35 0%, #B7D9B1 100%);
        color: #fff;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.92);
        border-radius: 14px;
        margin-bottom: 1.2rem;
        border: none;
    }
    /* Remove custom selected tab color, revert to Streamlit default */
    .stExpanderHeader {
        font-size: 1.1rem;
        color: #3A6B35;
    }
    .st-bb, .st-cq, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
        border-radius: 16px !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Enhanced question interface ---
def enhanced_question_interface():
    st.subheader("💬 Ask About Travel, Cultures, and Food")
    with st.expander("💡 Example questions you can ask"):
        st.write("""
        • What are the most famous dishes in [country/city]?
        • Describe unique cultural traditions in [place].
        • What landmarks should I visit in [destination]?
        • Compare the food culture between [place1] and [place2].
        • What festivals are celebrated in [country/city]?
        """)
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What are the best foods to try in Italy?"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        search_button = st.button("🔍 Search Documents", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear History")
    return question, search_button, clear_button

# --- Enhanced main function ---
def main():
    add_custom_css()
    st.markdown('<h1 class="main-header">🌍✈️ Travel, Cultures & Food Knowledge Hub 🍜🍕🥑🍣</h1>', unsafe_allow_html=True)
    st.markdown('<div class="main-header-spacer"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; font-size:1.2rem; margin-bottom:1.5rem;'>
        Welcome!<br>
        <b>Upload travel guides, cultural stories, and food adventures.<br>
        Ask questions, discover new places, and celebrate the world's diversity! 🌎🍲🕌🍉</b>
    </div>
    """, unsafe_allow_html=True)
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "❓ Ask Questions", "📋 Manage"])
    with tab1:
        st.header("Upload & Convert Travel, Culture, and Food Documents")
        uploaded_files = st.file_uploader(
            "Choose files (PDF, DOC, DOCX, TXT)",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True
        )
        if st.button("Convert & Add"):
            if uploaded_files:
                converted_docs = []
                errors = []
                for uploaded in uploaded_files:
                    file_ext = Path(uploaded.name).suffix.lower()
                    if len(uploaded.getvalue()) > 10 * 1024 * 1024:
                        errors.append(f"{uploaded.name}: File too large (max 10MB)")
                        continue
                    if file_ext not in ['.pdf', '.doc', '.docx', '.txt']:
                        errors.append(f"{uploaded.name}: Unsupported file type")
                        continue
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name
                    try:
                        md = convert_to_markdown(tmp_path)
                        if len(md.strip()) < 10:
                            errors.append(f"{uploaded.name}: File appears to be empty or corrupted")
                            continue
                        st.session_state.converted_docs.append({
                            "filename": uploaded.name,
                            "content": md
                        })
                        embedding = embed_text(md)
                        collection.add(
                            documents=[md],
                            embeddings=[embedding],
                            ids=[uploaded.name]
                        )
                        converted_docs.append({
                            'filename': uploaded.name,
                            'word_count': len(md.split())
                        })
                        st.success(f"Converted {uploaded.name} successfully.")
                    except Exception as e:
                        errors.append(f"{uploaded.name}: {str(e)}")
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                if converted_docs:
                    total_words = sum(doc['word_count'] for doc in converted_docs)
                    st.info(f"📊 Total words added: {total_words:,}")
                    with st.expander("📋 View converted files"):
                        for doc in converted_docs:
                            st.write(f"• **{doc['filename']}** - {doc['word_count']:,} words")
                if errors:
                    st.error(f"❌ {len(errors)} files failed to convert:")
                    for error in errors:
                        st.write(f"• {error}")
            else:
                st.warning("Please select files to upload first.")
    with tab2:
        st.header("Ask Questions About Travel, Cultures, and Food")
        if st.session_state.converted_docs:
            question, search_button, clear_button = enhanced_question_interface()
            if search_button and question:
                results = collection.query(
                    query_texts=[question],
                    n_results=1
                )
                if results["documents"]:
                    context = results["documents"][0][0]
                    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                    answer = qa_pipeline(prompt, max_length=150)[0]['generated_text'].strip()
                    st.markdown("### 💡 Answer")
                    st.write(answer)
                    st.info(f"📄 Source: {st.session_state.converted_docs[0]['filename']}")
                    add_to_search_history(question, answer, st.session_state.converted_docs[0]['filename'])
                else:
                    st.write("No answer found.")
            if clear_button:
                st.session_state.search_history = []
                st.success("Search history cleared!")
            if st.session_state.search_history:
                show_search_history()
        else:
            st.info("🔼 Upload some documents first to start asking questions!")
    with tab3:
        show_document_manager()
    with st.expander("About this Travel & Culture Q&A System"):
        st.write("""
        I created this app to answer questions about:
        - 🍲 Traditional foods and how they reflect culture
        - 🎉 Global festivals and their social meaning
        - 🗣️ Languages and worldviews
        - 🌱 Sustainable tourism practices
        - 🙏 Cultural etiquette around the world

        ✨ Try asking about specific dishes, customs, festivals, or etiquette rules in different countries!
        """)
    st.markdown("---")
    st.markdown("<div style='text-align:center; font-size:1.1rem;'>Made with ❤️ for explorers, foodies, and culture lovers!</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

