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
    origin = request.headers.get('Origin')
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://oohr-erp.web.app",
        "https://rag3-bfcu.onrender.com"  # ← ADD THIS LINE
    ]
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
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
    "Aries": [
        "Ajay Devgn", "Kapil Sharma", "Dr. A.P.J. Abdul Kalam",  # Missile Man of India
        "Emraan Hashmi", "Robert Downey Jr."
    ],
    "Taurus": [
        "Sachin Tendulkar", "Anushka Sharma", "G. D. Naidu",     # Indian Edison
        "Madhuri Dixit", "David Beckham"
    ],
    "Gemini": [
        "Sonam Kapoor", "Shilpa Shetty", "Karan Johar",
        "Dr. B. R. Ambedkar",  # Architect of Indian Constitution
        "Angelina Jolie"
    ],
    "Cancer": [
        "Priyanka Chopra", "MS Dhoni", "Ranveer Singh",
        "J. R. D. Tata",  # Industrialist and philanthropist
        "Ariana Grande"
    ],
    "Leo": [
        "Saif Ali Khan", "Sridevi", "Jacqueline Fernandez",
        "Bal Gangadhar Tilak",  # Freedom fighter
        "Barack Obama"
    ],
    "Virgo": [
        "Akshay Kumar", "Kareena Kapoor", "Narendra Modi",
        "Verghese Kurien",  # Father of White Revolution
        "Michael Jackson"
    ],
    "Libra": [
        "Amitabh Bachchan", "Rekha", "Ranbir Kapoor",
        "Dr. Vikram Sarabhai",  # Father of Indian Space Program
        "Will Smith"
    ],
    "Scorpio": [
        "Shah Rukh Khan", "Aishwarya Rai", "Sushmita Sen",
        "Lal Bahadur Shastri",  # Former PM
        "Bill Gates"
    ],
    "Sagittarius": [
        "Yami Gautam", "Dharmendra", "John Abraham",
        "Kalpana Chawla",  # Astronaut
        "Taylor Swift"
    ],
    "Capricorn": [
        "Deepika Padukone", "Hrithik Roshan", "Javed Akhtar",
        "Swami Vivekananda",  # Spiritual leader
        "Michelle Obama"
    ],
    "Aquarius": [
        "Preity Zinta", "Abhishek Bachchan", "Jackie Shroff",
        "Ratan Tata",  # Business leader
        "Oprah Winfrey"
    ],
    "Pisces": [
        "Alia Bhatt", "Shahid Kapoor", "Tiger Shroff",
        "C. V. Raman",  # Physicist & Nobel laureate
        "Albert Einstein"
    ]
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

    # 🧹 Clean any unwanted lines at the start
    filtered_lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Skip irrelevant or fallback lines
        if "I don't have the capability" in line:
            continue
        if "I’m sorry" in line:
            continue
        filtered_lines.append(line)

    # 🔍 Now parse filtered lines
    for line in filtered_lines:
        if not line:
            continue
        if "strength" in line.lower():
            current_section = "strengths"
            continue
        elif "weakness" in line.lower():
            current_section = "weaknesses"
            continue
        elif "recommendation" in line.lower():
            current_section = "recommendations"
            continue

        if current_section and not line.startswith(('###', '---')):
            formatted_line = format_response_item(line)
            sections[current_section].append(formatted_line)

    # Make sure each section has exactly 3 items
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

🧠 Basic Information:
- Date of Birth: {dob}
- Time of Birth: {time_of_birth}
- Place of Birth: {place_of_birth}
- Zodiac Sign: {zodiac}
- Famous People with Same Sign: {', '.join(famous_people)}

🧩 Psychological Traits (DSM-5 indicators):
{', '.join(symptoms)}

📘 Academic Performance Summary:
{academic_summary if academic_summary else "Academic records were not provided."}

📊 Based on the child's **astrological sign ({zodiac})**, psychological traits, and academic records, please provide a report with:

1. **Three Key Strengths**  
   - Integrate astrological, psychological, and academic strengths  
   - Highlight subject-specific performance if available  

2. **Three Areas for Improvement**  
   - Include academic gaps if reflected in the records  
   - Consider emotional or behavioral weaknesses  

3. **Three Personalized Recommendations**  
   - Realistic suggestions based on child’s learning profile  
   - Can include educational strategies, emotional support, or extracurriculars  

📌 Also, include this note at the end:  
_"This report will be taken again during training to improve accuracy and provide more refined insights."_

💡 Notes:  
- Bold important traits or categories (**like this**)  
- Connect zodiac personality traits to any behavioral/learning patterns  
- Maintain a clear balance between astrology, psychology, and academics  
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
