from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Allow all origins temporarily for testing
from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Step 3: Load API key from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Constants
PDF_FILES = [
    "./allknowledgebase/DSM-5-TR.pdf",
    "./allknowledgebase/Astrology For The Soul PDF.pdf",
    "./allknowledgebase/Numerology and the Divine Triangle (Faith Javane).pdf",
    "./allknowledgebase/the-only-astrology-book-youll-ever-need_compress.pdf"
    "./allknowledgebase/ACEDEMICSKB.pdf",
]
FAISS_INDEX_PATH = "faiss_index"

# Initialize Flask app


# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Utility: Load and chunk multiple PDFs
def load_and_split_documents(file_paths):
    docs = []
    for file in file_paths:
        if os.path.exists(file):
            print(f"✅ Loading: {file}")
            loader = PyMuPDFLoader(file)
            loaded = loader.load()
            print(f"  --> Loaded {len(loaded)} pages")
            docs.extend(loaded)
        else:
            print(f"❌ File not found: {file}")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"🧩 Total chunks created: {len(chunks)}")
    return chunks


# Load or build FAISS vector store
def get_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        print("📁 Loading existing FAISS index...")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("📄 Building new FAISS index...")
        chunks = load_and_split_documents(PDF_FILES)

        batch_size = 100
        vectorstore = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"🔄 Processing batch {i // batch_size + 1}...")
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)

        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore


# Initialize RAG components
vector_store = get_vector_store()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)
from datetime import datetime

# Zodiac mapping logic
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
def analyze_academic_pattern(percentages):
    if not percentages or len(percentages) < 3:
        return "Academic pattern: Not enough data.", [], []

    avg = sum(percentages) / len(percentages)
    trend = "flat"
    if percentages[-1] > percentages[0]:
        if percentages[-1] - percentages[0] >= 10:
            trend = "improving"
    elif percentages[0] - percentages[-1] >= 10:
        trend = "declining"

    fluctuation = max(percentages) - min(percentages)

    # Match against KB patterns
    if avg >= 75:
        pattern = "consistent_performer"
        tags = ["academic_consistency", "self_motivation"]
        recs = [
            "Encourage participation in enrichment programs.",
            "Consider leadership roles in academic teams."
        ]
    elif avg <= 60:
        pattern = "underperforming"
        tags = ["academic_struggle", "possible_learning_difficulty"]
        recs = [
            "Recommend diagnostic academic or psychological assessment.",
            "Provide remedial support and mentoring interventions."
        ]
    elif trend == "declining":
        pattern = "declining_trend"
        tags = ["academic_decline", "possible_attention_issues"]
        recs = [
            "Evaluate for potential attention or emotional factors.",
            "Schedule regular academic reviews with guardians."
        ]
    elif trend == "improving":
        pattern = "improving_trend"
        tags = ["positive_academic_growth", "resilience"]
        recs = [
            "Reinforce study habits and motivation strategies.",
            "Recognize and reward progress to maintain momentum."
        ]
    elif fluctuation >= 15:
        pattern = "inconsistent_performance"
        tags = ["inconsistent_focus", "possible_environmental_factors"]
        recs = [
            "Investigate stability of support systems at home or school.",
            "Implement a consistent academic routine."
        ]
    else:
        pattern = "average_performance"
        tags = []
        recs = []
    description = f"Academic pattern: {pattern.replace('_', ' ').title()} (Avg: {avg:.1f}%, Trend: {trend}, Fluctuation: {fluctuation}%)"
    return description, tags, recs


# Routes
@app.route("/rag", methods=["POST"])
def rag():
    data = request.json
    dob = data.get("dob", "Not provided")
    time_of_birth = data.get("time_of_birth", "Not provided")
    place_of_birth = data.get("place_of_birth", "Not provided")
    symptoms = data.get("symptom_keywords", [])

    academic_info = data.get("academic_info", {})
    percentages = academic_info.get("percentages_last_3_years", [])
    academic_desc, academic_tags, academic_recs = analyze_academic_pattern(percentages)

    zodiac, famous_people = get_zodiac_and_famous_people(dob)

    query = f"""
A child was born on {dob} at {time_of_birth} in {place_of_birth}.
Zodiac sign: {zodiac}. Famous people with this sign: {', '.join(famous_people)}.
The child's psychological traits include: {', '.join(symptoms)}.
{academic_desc}
Academic recommendations: {', '.join(academic_recs)}

Using DSM-5 psychological criteria, astrological traits, and academic/educational theory from the knowledge base, provide a holistic, non-clinical exploration of this child's strengths, weaknesses, and growth opportunities.

For the academic perspective, analyze the provided academic pattern and recommendations, and blend them with psychological and astrological insights.

Do not provide a clinical diagnosis. Present the analysis as a blend of psychological, astrological, and academic insights for educational purposes only.

List:
1. 3 strengths (showing how psychological, astrological, and academic perspectives reinforce or contrast each other)
2. 3 weaknesses or risk areas (with similar commentary)
3. 3 recommendations for growth (drawing from all three perspectives)
"""

    result = qa_chain({"query": query})
    full_answer = result["result"]

    # Basic parsing for strengths, weaknesses, recommendations
    def extract_section(text, header):
        lines = text.splitlines()
        found = []
        collect = False
        for line in lines:
            if header.lower() in line.lower():
                collect = True
                continue
            if collect:
                if line.strip() == "" or line.strip().lower().startswith("recommendation") or line.strip().lower().startswith("weakness") or line.strip().lower().startswith("strength"):
                    break
                found.append(line.strip("-•: \t"))
        return [line for line in found if line]

    strengths = extract_section(full_answer, "Strength")
    weaknesses = extract_section(full_answer, "Weakness")
    recommendations = extract_section(full_answer, "Recommendation")

    return jsonify({
        "strengths": strengths or ["Strength not parsed."],
        "weaknesses": weaknesses or ["Weakness not parsed."],
        "recommendations": recommendations or ["Recommendation not parsed."],
        "zodiac": zodiac,
        "famous_people": famous_people,
        "academic_pattern": academic_desc,
        "academic_tags": academic_tags,
        "academic_recommendations": academic_recs,
        "raw_answer": full_answer
    })
if __name__ == "__main__":
   app.run(debug=True, host="127.0.0.1", port=5000)


@app.route("/astrology", methods=["POST"])
def astrology():
    data = request.json
    dob = data.get("dob")
    time_of_birth = data.get("time_of_birth")
    place_of_birth = data.get("place_of_birth")

    zodiac, famous_people = get_zodiac_and_famous_people(dob)

    query = f"""
    A child was born on {dob} at {time_of_birth} in {place_of_birth}.
    Zodiac sign: {zodiac}. Famous people with this sign: {', '.join(famous_people)}.
    Provide 3 strengths, 3 weaknesses, and 3 growth suggestions based on astrology and age traits.
    """
    result = qa_chain({"query": query})

    return jsonify({
        "zodiac": zodiac,
        "famous_people": famous_people,
        "answer": result["result"],
    })
@app.route("/psychology", methods=["POST"])
def psychology():
    data = request.json
    symptoms = data.get("symptom_keywords", [])

    query = f"""
    The child is showing these traits: {', '.join(symptoms)}.
    Analyze their mental health using DSM-5 guidelines.
    Provide 3 strengths, 3 weaknesses or risks, and 3 recommendations.
    """
    result = qa_chain({"query": query})

    return jsonify({
        "answer": result["result"]
    })
@app.route("/final-analysis", methods=["POST"])
def final_analysis():
    data = request.json
    astro = data.get("astrology", {})
    psych = data.get("psychology", {})
    academics = data.get("academics", {})

    query = f"""
    Based on the following:
    Astrology Info: {astro.get('answer')}
    Mental Health Traits: {psych.get('answer')}
    Academic Data: Grades: {academics.get('grades')}, Comments: {academics.get('comments')}

    Create a final comprehensive diagnostic report with:
    1. Overview Summary
    2. Key Strengths
    3. Areas for Growth
    4. Tailored Recommendations
    """
    result = qa_chain({"query": query})

    return jsonify({
        "summary": result["result"]
    })

