from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from datetime import datetime
import re

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "https://oohr-erp.web.app"  # ← replace with your real frontend URL
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Constants
PDF_FILES = [
    "./allknowledgebase/DSM-5-TR.pdf",
    "./allknowledgebase/Astrology For The Soul PDF.pdf",
    "./allknowledgebase/Numerology and the Divine Triangle (Faith Javane).pdf",
    "./allknowledgebase/the-only-astrology-book-youll-ever-need_compress.pdf"
]
FAISS_INDEX_PATH = "faiss_index"

# Initialize AI components
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Zodiac configuration
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

FAMOUS_ZODIACS = {
    "Aries": ["Lady Gaga", "Robert Downey Jr."],
    "Taurus": ["Gigi Hadid", "David Beckham"],
    "Gemini": ["Angelina Jolie", "Kanye West"],
    "Cancer": ["Elon Musk", "Ariana Grande"],
    "Leo": ["Barack Obama", "Jennifer Lopez"],
    "Virgo": ["Beyoncé", "Michael Jackson", "Warren Buffett"],
    "Libra": ["Will Smith", "Kim Kardashian"],
    "Scorpio": ["Bill Gates", "Drake"],
    "Sagittarius": ["Taylor Swift", "Brad Pitt"],
    "Capricorn": ["Michelle Obama", "Denzel Washington"],
    "Aquarius": ["Oprah Winfrey", "Harry Styles"],
    "Pisces": ["Rihanna", "Albert Einstein"],
}

def get_zodiac_and_famous_people(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        month, day = dob.month, dob.day
        for sign, start, end in ZODIAC_SIGNS:
            if (month, day) >= start and (month, day) <= end:
                return sign, FAMOUS_ZODIACS.get(sign, [])
    except Exception:
        pass
    return "Unknown", []

def format_response_item(item):
    """Ensure consistent bold formatting in response items"""
    if not isinstance(item, str):
        return item
    # Add bold formatting to key terms if missing
    key_terms = [
        "Social Engagement", "Self-Efficacy", "Temperament", 
        "Internalizing", "Self-Esteem", "School Refusal",
        "Emotional Expression", "Dependent Behavior", 
        "Parental Reinforcement", "Communication",
        "Independence", "Social Interaction"
    ]
    for term in key_terms:
        if term in item and f"**{term}**" not in item:
            item = item.replace(term, f"**{term}**")
    return item

def parse_report_sections(text):
    """Improved parsing of the AI response into structured sections"""
    sections = {
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }
    current_section = None
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Detect section headers
        if "strength" in line.lower():
            current_section = "strengths"
            continue
        elif "weakness" in line.lower():
            current_section = "weaknesses"
            continue
        elif "recommendation" in line.lower():
            current_section = "recommendations"
            continue
        
        # Add content to current section
        if current_section and line and not line.startswith(('###', '---')):
            formatted_line = format_response_item(line)
            sections[current_section].append(formatted_line)
    
    # Ensure exactly 3 items per section
    for section in sections:
        sections[section] = sections[section][:3]
        if not sections[section]:
            sections[section] = [f"No {section} identified"]
    
    return sections

@app.route('/rag', methods=['OPTIONS'])
def handle_options():
    return jsonify({'message': 'Preflight request accepted'}), 200

@app.route("/rag", methods=["POST"])
def rag():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Validate required fields
        required_fields = ['dob', 'time_of_birth', 'place_of_birth', 'symptom_keywords']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        dob = data['dob']
        time_of_birth = data['time_of_birth']
        place_of_birth = data['place_of_birth']
        symptoms = data['symptom_keywords']
        academic_records = data.get('academic_records', [])

        # Get zodiac information
        zodiac, famous_people = get_zodiac_and_famous_people(dob)

        # Prepare academic summary if available
        academic_summary = ""
        if academic_records:
            academic_summary = "\nAcademic Performance:\n" + "\n".join(
                f"{rec.get('year', '')} - Class {rec.get('class', '')}: " +
                ", ".join(f"{sub['subject']} ({sub['percentage']}%)" 
                for sub in rec.get('subjects', []))
                for rec in academic_records
            )

        # Construct the standardized query
        query = f"""
        Comprehensive Child Profile Analysis Request:
        
        Basic Information:
        - Date of Birth: {dob}
        - Time of Birth: {time_of_birth}
        - Place of Birth: {place_of_birth}
        - Zodiac Sign: {zodiac}
        - Famous People with Same Sign: {', '.join(famous_people)}
        
        Psychological Traits:
        {', '.join(symptoms)}
        
        {academic_summary}
        
        Please provide a detailed analysis with exactly:
        1. Three key strengths (bold each strength category with **)
        2. Three areas for improvement (bold each weakness with **)
        3. Three specific recommendations (bold each recommendation focus with **)
        
        Format each section clearly with numbered items and maintain consistent 
        bold formatting for key psychological terms throughout the report.
        """

        # Get AI response
        result = qa_chain({"query": query})
        full_answer = result["result"]

        # Parse the response into structured sections
        sections = parse_report_sections(full_answer)

        # Prepare the standardized response
        response_data = {
            "strengths": sections["strengths"],
            "weaknesses": sections["weaknesses"],
            "recommendations": sections["recommendations"],
            "zodiac": zodiac,
            "famous_people": famous_people,
            "raw_answer": full_answer
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in /rag endpoint: {str(e)}")
        return jsonify({
            "error": "Failed to generate report",
            "details": str(e)
        }), 500
    
@app.route("/", methods=["GET"])
def home():
    return "✅ Flask RAG API is live and ready!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
