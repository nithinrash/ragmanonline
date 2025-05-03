from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
import os
from werkzeug.utils import secure_filename
import logging
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import time

# --- Flask setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Global settings ---
EMBED_MODEL = "sentence-transformers/all-distilroberta-v1"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
LLM_PATH = "models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
ALLOWED_EXTENSIONS = {'pdf'}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

# --- Load embedding model ---
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# --- Global variables ---
qa_chain = None
chunks_with_page_numbers = []

# --- Helper function to validate file type ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- PDF Processing and VectorStore setup ---
def process_pdf(path):
    global qa_chain, chunks_with_page_numbers

    try:
        # Load document
        loader = PyMuPDFLoader(path)
        documents = loader.load()

        # Split into chunks with page numbers
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks_with_page_numbers = []

        # Iterate through documents and manually add page numbers
        for page_num, doc in enumerate(documents, start=1):
            chunks = splitter.split_documents([doc])
            for chunk in chunks:
                chunks_with_page_numbers.append((chunk, page_num))

        # Embed and store in FAISS
        db = FAISS.from_documents([chunk[0] for chunk in chunks_with_page_numbers], embedding_model)

        # Initialize LLM
        llm = LlamaCpp(
            model_path=LLM_PATH,
            n_ctx=8192,
            n_threads=4,
            n_batch=512,
            temperature=0.2,
            max_tokens=150,
            stop=["Question:"]
        )

        # Setup retrieval QA chain with updated vector store
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        raise

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['pdf']
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            process_pdf(filepath)
            return redirect(url_for('qa'))
        except Exception as e:
            return jsonify({"error": f"Error processing file: {e}"}), 500
    return jsonify({"error": "Invalid file format. Only PDF files are allowed."}), 400

@app.route('/qa', methods=['GET', 'POST'])
def qa():
    if request.method == 'POST':
        question = request.form['question']
        if qa_chain:
            def generate_answer():
                answer = qa_chain.run(question)
                chunks = [f"Page {page_number}:\n{chunk}" for chunk, page_number in chunks_with_page_numbers]

                # Stream text one by one (simulate a chat-like experience)
                for chunk in chunks:
                    for word in chunk.split(" "):
                        yield word + " "
                        time.sleep(0.05)  # Adjust this to control the speed of streaming
                    yield "\n"
                yield answer
            return Response(generate_answer(), content_type='text/plain')
        return render_template('qa.html', question=question, answer="No document processed.")
    return render_template('qa.html')

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8000, threaded=True)
