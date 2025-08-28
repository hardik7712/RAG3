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
            "https://nurture-spark-portal.lovable.app"
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

FAMOUS_ZODIACS = {
    "Aries": ["Ajay Devgn", "Kapil Sharma", "Dr. A.P.J. Abdul Kalam", "Emraan Hashmi", "Robert Downey Jr."],
    "Taurus": ["Sachin Tendulkar", "Anushka Sharma", "G. D. Naidu", "Madhuri Dixit", "David Beckham"],
    "Gemini": ["Sonam Kapoor", "Shilpa Shetty", "Karan Johar", "Dr. B. R. Ambedkar", "Angelina Jolie"],
    "Cancer": ["Priyanka Chopra", "MS Dhoni", "Ranveer Singh", "J. R. D. Tata", "Ariana Grande"],
    "Leo": ["Saif Ali Khan", "Sridevi", "Jacqueline Fernandez", "Bal Gangadhar Tilak", "Barack Obama"],
    "Virgo": ["Akshay Kumar", "Kareena Kapoor", "Narendra Modi", "Verghese Kurien", "Michael Jackson"],
    "Libra": ["Amitabh Bachchan", "Rekha", "Ranbir Kapoor", "Dr. Vikram Sarabhai", "Will Smith"],
    "Scorpio": ["Shah Rukh Khan", "Aishwarya Rai", "Sushmita Sen", "Lal Bahadur Shastri", "Bill Gates"],
    "Sagittarius": ["Yami Gautam", "Dharmendra", "John Abraham", "Kalpana Chawla", "Taylor Swift"],
    "Capricorn": ["Deepika Padukone", "Hrithik Roshan", "Javed Akhtar", "Swami Vivekananda", "Michelle Obama"],
    "Aquarius": ["Preity Zinta", "Abhishek Bachchan", "Jackie Shroff", "Ratan Tata", "Oprah Winfrey"],
    "Pisces": ["Alia Bhatt", "Shahid Kapoor", "Tiger Shroff", "C. V. Raman", "Albert Einstein"]
}

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
def format_response_item(item):
    if not isinstance(item, str):
        return item
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
    sections = {"strengths": [], "weaknesses": [], "recommendations": []}
    current_section = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "strength" in line.lower(): current_section = "strengths"; continue
        if "weakness" in line.lower(): current_section = "weaknesses"; continue
        if "recommendation" in line.lower(): current_section = "recommendations"; continue
        if current_section: sections[current_section].append(format_response_item(line))
    for k in sections:
        sections[k] = sections[k][:3]
        if not sections[k]: sections[k] = [f"No {k} identified"]
    return sections

# -------------------- Routes --------------------
@app.route('/rag', methods=['OPTIONS'])
def handle_options():
    return jsonify({'message': 'Preflight request accepted'}), 200

@app.route("/rag", methods=["POST"])
def rag():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Required fields
        for field in ['dob', 'time_of_birth', 'place_of_birth', 'symptom_keywords']:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        dob = data['dob']
        time_of_birth = data['time_of_birth']
        place_of_birth = data['place_of_birth']

        # Convert symptom_keywords to list if it's an object
        symptoms = data['symptom_keywords']
        if isinstance(symptoms, dict):
            symptoms = list(symptoms.values())
        elif not isinstance(symptoms, list):
            symptoms = []

        academic_records = data.get('academic_records', [])

        # Zodiac
        zodiac, famous_people = get_zodiac_and_famous_people(dob)

        # Academic summary
        academic_summary = ""
        if isinstance(academic_records, str):
            academic_summary = f"\nAcademic Performance:\n{academic_records}"
        elif isinstance(academic_records, list):
            # optional structured format
            academic_summary = "\nAcademic Performance:\n" + "\n".join(
                f"{rec.get('year','')} - Class {rec.get('class','')}: " +
                ", ".join(f"{sub['subject']} ({sub['percentage']}%)" 
                for sub in rec.get('subjects',[]))
                for rec in academic_records
            )

        # Construct query
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

📊 Please provide:
1. Three Key Strengths
2. Three Areas for Improvement
3. Three Personalized Recommendations

💡 Notes:
- Bold important traits (**like this**)
"""

        # Call QA chain
        result = qa_chain({"query": query})
        full_answer = result.get("result", "No AI response generated.")

        sections = parse_report_sections(full_answer)

        return jsonify({
            "strengths": sections["strengths"],
            "weaknesses": sections["weaknesses"],
            "recommendations": sections["recommendations"],
            "zodiac": zodiac,
            "famous_people": famous_people,
            "raw_answer": full_answer
        })

    except Exception as e:
        print("Error in /rag endpoint:", e, flush=True)
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask RAG API is live and ready!"

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
