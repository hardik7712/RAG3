from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from datetime import datetime

# -------------------- Flask App & CORS --------------------
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "",
            "http://192.168.56.1:8080",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5000",
            "http://127.0.0.1:5000",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://oohr-erp.web.app",
        "https://rag3-bfcu.onrender.com"
    ]
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# -------------------- Environment --------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -------------------- AI Components --------------------
FAISS_INDEX_PATH = "faiss_index"
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# -------------------- Zodiac --------------------
ZODIAC_SIGNS = [
    ("Capricorn", (1, 1), (1, 19)),
    ("Aquarius", (1, 20), (2, 18)),
    ("Pisces", (2, 19), (3, 20)),
    ("Aries", (3, 21), (4, 19)),
    ("Taurus", (4, 20), (5, 20)),
    ("Gemini", (5, 21), (6, 20)),
    ("Cancer", (6, 21), (7, 22)),
    ("Leo", (7, 23), (8, 22)),
    ("Virgo", (8, 23), (9, 22)),
    ("Libra", (9, 23), (10, 22)),
    ("Scorpio", (10, 23), (11, 21)),
    ("Sagittarius", (11, 22), (12, 21)),
    ("Capricorn", (12, 22), (12, 31)),
]

FAMOUS_ZODIACS = { ... }  # unchanged dictionary

def get_zodiac_and_famous_people(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        month, day = dob.month, dob.day
        for sign, start, end in ZODIAC_SIGNS:
            if (month, day) >= start and (month, day) <= end:
                return sign, FAMOUS_ZODIACS.get(sign, [])
    except Exception as e:
        print("Zodiac parsing error:", e)
    return "Unknown", []

# -------------------- Helpers --------------------
def format_response_item(item): ...
def parse_report_sections(text): ...

# -------------------- Routes --------------------
@app.route('/rag', methods=['OPTIONS'])
def handle_options():
    return jsonify({'message': 'Preflight request accepted'}), 200

@app.route("/rag", methods=["POST"])
def rag():
    # 🔹 existing RAG route (unchanged)
    ...

# 🔹 NEW ROUTE: Accepts a JSON { "message": "...", "personal_info": {...} }
@app.route("/rag-message", methods=["POST"])
def rag_message():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        message = data.get("message")
        personal_info = data.get("personal_info", {})

        if not message:
            return jsonify({"error": "Missing required field: message"}), 400

        # Convert personal_info dict into a readable string
        personal_info_str = "\n".join([f"{k}: {v}" for k, v in personal_info.items()])

        # Construct query for RAG
        query = f"""
User Message:
{message}

Personal Information Provided:
{personal_info_str if personal_info else "No personal info provided."}

Please generate a single comprehensive reply based on the above, using the knowledge base.
"""

        result = qa_chain({"query": query})
        full_answer = result.get("result", "No AI response generated.")

        return jsonify({
            "reply": full_answer
        })

    except Exception as e:
        print("Error in /rag-message endpoint:", e, flush=True)
        return jsonify({"error": "Failed to process message", "details": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "✅ Flask RAG API is live and ready!"

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
