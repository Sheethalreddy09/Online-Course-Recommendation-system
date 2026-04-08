from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tensorflow as tf
import keras
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from groq import Groq
import os
import re
import json
import tempfile
import zipfile

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

load_dotenv()

app = FastAPI(
    title="Online Course Recommendation API",
    description="Personalized course recommendations using Deep Learning + TF-IDF Retrieval",
    version="3.0.0"
)

# =====================================================
# CORS — allow Streamlit frontend to call FastAPI
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Groq LLM Client
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

MODEL_PATH = os.path.join(MODEL_DIR, "model2.keras")
USER_ENCODER_PATH = os.path.join(MODEL_DIR, "user_encoder.pkl")
COURSE_ENCODER_PATH = os.path.join(MODEL_DIR, "course_encoder.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")

PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_online_course_data.csv")
COURSE_KB_PATH = os.path.join(DATA_DIR, "course_knowledge_base.csv")
NEW_USERS_PATH = os.path.join(DATA_DIR, "new_users.csv")

os.makedirs(DATA_DIR, exist_ok=True)


def _remove_unsupported_keras_fields(obj):
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            # Some saved Keras model archives include fields that older/newer
            # loaders reject even when they are null. Strip them before loading.
            if key == "quantization_config":
                continue
            cleaned[key] = _remove_unsupported_keras_fields(value)
        return cleaned

    if isinstance(obj, list):
        return [_remove_unsupported_keras_fields(item) for item in obj]

    return obj


def load_compatible_keras_model(model_path: str):
    with zipfile.ZipFile(model_path, "r") as src_zip:
        config = json.loads(src_zip.read("config.json"))
        cleaned_config = _remove_unsupported_keras_fields(config)

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
            temp_model_path = temp_file.name

        with zipfile.ZipFile(temp_model_path, "w") as dst_zip:
            for info in src_zip.infolist():
                if info.filename == "config.json":
                    dst_zip.writestr(info, json.dumps(cleaned_config))
                else:
                    dst_zip.writestr(info, src_zip.read(info.filename))

    try:
        return keras.models.load_model(temp_model_path, compile=False)
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

# =====================================================
# Load Artifacts
# =====================================================
print("Loading deep learning model...")
model = load_compatible_keras_model(MODEL_PATH)

print("Loading encoders...")
with open(USER_ENCODER_PATH, "rb") as f:
    user_encoder = pickle.load(f)

with open(COURSE_ENCODER_PATH, "rb") as f:
    course_encoder = pickle.load(f)

print("Loading TF-IDF retriever...")
with open(TFIDF_VECTORIZER_PATH, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(TFIDF_MATRIX_PATH, "rb") as f:
    tfidf_matrix = pickle.load(f)

print("Loading datasets...")
df = pd.read_csv(PROCESSED_DATA_PATH)
course_kb = pd.read_csv(COURSE_KB_PATH)

df["user_id"] = df["user_id"].astype(int)
df["user_encoded"] = df["user_encoded"].astype(int)
df["course_encoded"] = df["course_encoded"].astype(int)

NEW_USERS_PATH = os.path.join(DATA_DIR, "new_users.csv")

expected_columns = ["user_id", "course_name", "instructor", "difficulty_level", "rating"]

if os.path.exists(NEW_USERS_PATH):
    new_users_df = pd.read_csv(NEW_USERS_PATH)

    # If the CSV has the same columns in a different order, keep the data and reorder it.
    # Recreate only when the schema is genuinely different.
    if set(new_users_df.columns) != set(expected_columns):
        print("Old new_users.csv schema detected. Recreating with new schema...")
        new_users_df = pd.DataFrame(columns=expected_columns)
        new_users_df.to_csv(NEW_USERS_PATH, index=False)
    else:
        new_users_df = new_users_df[expected_columns]
        if not new_users_df.empty:
            new_users_df["user_id"] = new_users_df["user_id"].astype(int)
            new_users_df["course_name"] = new_users_df["course_name"].astype(str)
            new_users_df["instructor"] = new_users_df["instructor"].astype(str)
            new_users_df["difficulty_level"] = new_users_df["difficulty_level"].astype(str)
            new_users_df["rating"] = new_users_df["rating"].astype(float)
else:
    new_users_df = pd.DataFrame(columns=expected_columns)
    new_users_df.to_csv(NEW_USERS_PATH, index=False)


print("✅ All artifacts loaded successfully!")
print("Main df shape:", df.shape)
print("Course KB shape:", course_kb.shape)
print("New users shape:", new_users_df.shape)

# =====================================================
# Request Schemas
# =====================================================
class RecommendRequest(BaseModel):
    user_id: int

class SearchRequest(BaseModel):
    query: str

class UserSuggestRequest(BaseModel):
    prefix: str

class CourseSuggestRequest(BaseModel):
    prefix: str

class CreateNewUserRequest(BaseModel):
    course_name: str
    instructor: str
    difficulty_level: str
    rating: float

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
    history: Optional[list[ChatMessage]] = []
    all_retrieved_courses: Optional[list[dict]] = []

# =====================================================
# Alias Map
# =====================================================
COURSE_NAMES = course_kb["course_name"].dropna().unique().tolist()

COURSE_ALIASES = {
    "AI for Business Leaders": [
        "ai", "artificial intelligence", "ai business", "business ai"
    ],
    "Advanced Machine Learning": [
        "ml", "machine learning", "advanced ml", "machine learning course", "machine"
    ],
    "Blockchain and Decentralized Applications": [
        "blockchain", "decentralized", "web3", "crypto applications"
    ],
    "Cloud Computing Essentials": [
        "cloud", "cloud computing", "aws", "azure", "gcp"
    ],
    "Cybersecurity for Professionals": [
        "cybersecurity", "cyber security", "security", "infosec", "cyber"
    ],
    "Data Visualization with Tableau": [
        "data visualization", "tableau", "dashboard", "visualization"
    ],
    "DevOps and Continuous Deployment": [
        "devops", "ci cd", "continuous deployment", "deployment"
    ],
    "Ethical Hacking Masterclass": [
        "ethical hacking", "hacking", "penetration testing", "pentest"
    ],
    "Fitness and Nutrition Coaching": [
        "fitness", "nutrition", "diet", "health coaching"
    ],
    "Fundamentals of Digital Marketing": [
        "digital marketing", "marketing", "seo", "social media marketing"
    ],
    "Game Development with Unity": [
        "game development", "unity", "game dev"
    ],
    "Graphic Design with Canva": [
        "graphic design", "canva", "design"
    ],
    "Mobile App Development with Swift": [
        "mobile app", "swift", "ios", "app development"
    ],
    "Networking and System Administration": [
        "networking", "system admin", "system administration", "network"
    ],
    "Personal Finance and Wealth Building": [
        "finance", "personal finance", "wealth", "investing basics"
    ],
    "Photography and Video Editing": [
        "photo", "photos", "photography", "picture", "pictures", "video editing"
    ],
    "Project Management Fundamentals": [
        "project management", "management", "pmp"
    ],
    "Public Speaking Mastery": [
        "public speaking", "speaking", "presentation skills"
    ],
    "Python for Beginners": [
        "python", "python course", "beginner python", "coding python"
    ],
    "Stock Market and Trading Strategies": [
        "stock market", "trading", "stocks", "investment trading"
    ],
}

# =====================================================
# Helpers
# =====================================================
def normalize_query(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    replacements = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "ds": "data science",
        "cyber security": "cybersecurity",
        "photos": "photography",
        "photo": "photography",
        "pics": "photography",
        "pic": "photography",
        "picture": "photography",
        "pictures": "photography",
        "apps": "app",
    }

    words = text.split()
    words = [replacements.get(w, w) for w in words]
    return " ".join(words)

def get_next_new_user_id():
    base_max = int(df["user_id"].max())
    if new_users_df.empty:
        return base_max + 1
    return max(base_max, int(new_users_df["user_id"].max())) + 1

def extract_difficulty(query: str):
    """Detect difficulty level keywords in the query."""
    q = query.lower()
    if any(w in q for w in ["advanced", "expert", "hard", "difficult"]):
        return "Advanced"
    if any(w in q for w in ["intermediate", "medium", "moderate"]):
        return "Intermediate"
    if any(w in q for w in ["beginner", "basic", "starter", "easy", "intro", "introduction"]):
        return "Beginner"
    return None

def retrieve_courses(query: str, top_n: int = 5):
    normalized = normalize_query(query)

    # Step 1: Detect difficulty level in query
    difficulty_filter = extract_difficulty(query)

    # Step 2: Work on filtered subset if difficulty is mentioned
    if difficulty_filter:
        filtered_kb = course_kb[course_kb["difficulty_level"] == difficulty_filter].reset_index(drop=True)
        if filtered_kb.empty:
            filtered_kb = course_kb.copy().reset_index(drop=True)
    else:
        filtered_kb = course_kb.copy().reset_index(drop=True)

    # Step 3: Rebuild TF-IDF on filtered subset
    filtered_docs = filtered_kb["course_document"].tolist()
    filtered_matrix = tfidf_vectorizer.transform(filtered_docs)
    query_vec = tfidf_vectorizer.transform([normalized])
    sim_scores = cosine_similarity(query_vec, filtered_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]

    results = filtered_kb.iloc[top_indices][[
        "course_name",
        "instructor",
        "difficulty_level",
        "certification_offered",
        "study_material_available",
        "course_price",
        "feedback_score",
        "rating"
    ]].copy()

    results["similarity_score"] = sim_scores[top_indices].round(4)
    return results.reset_index(drop=True).to_dict(orient="records")

def recommend_top_courses_for_user(model_obj, user_id_value: int, df_data: pd.DataFrame, top_n: int = 5):
    user_row = df_data[df_data["user_id"] == user_id_value]
    if user_row.empty:
        return None

    user_encoded_value = int(user_row["user_encoded"].iloc[0])

    taken_courses = set(df_data[df_data["user_id"] == user_id_value]["course_name"])

    candidate_courses = df_data[~df_data["course_name"].isin(taken_courses)][[
        "course_key",
        "course_name",
        "instructor",
        "difficulty_level",
        "course_encoded",
        "certification_offered_encoded",
        "study_material_available_encoded",
        "course_price_scaled",
        "feedback_score_scaled"
    ]].drop_duplicates(subset=["course_key"]).copy()

    if candidate_courses.empty:
        return []

    candidate_courses["user_encoded"] = user_encoded_value

    predictions = model_obj.predict(
        [
            candidate_courses["user_encoded"].values.reshape(-1, 1),
            candidate_courses["course_encoded"].values.reshape(-1, 1),
            candidate_courses[[
                "certification_offered_encoded",
                "study_material_available_encoded",
                "course_price_scaled",
                "feedback_score_scaled"
            ]].values.astype("float32")
        ],
        verbose=0
    )

    candidate_courses["predicted_rating"] = predictions.flatten()

    top_courses = candidate_courses.sort_values(
        "predicted_rating", ascending=False
    ).drop_duplicates(subset=["course_name"]).head(top_n)

    return (
        top_courses[[
            "course_name",
            "instructor",
            "difficulty_level",
            "predicted_rating"
        ]]
        .round({"predicted_rating": 4})
        .to_dict(orient="records")
    )

def recommend_related_from_selected_course(course_name: str, top_n: int = 5):
    query_vec = tfidf_vectorizer.transform([normalize_query(course_name)])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1]

    results = course_kb.iloc[top_indices][[
        "course_name",
        "instructor",
        "difficulty_level",
        "certification_offered",
        "study_material_available",
        "course_price",
        "feedback_score",
        "rating"
    ]].copy()

    results["similarity_score"] = sim_scores[top_indices].round(4)
    results = results[results["course_name"].str.lower() != course_name.lower()]
    results = results.sort_values(
        by=["similarity_score", "rating", "feedback_score"],
        ascending=[False, False, False]
    )

    return results.head(top_n).reset_index(drop=True).to_dict(orient="records")

def find_best_matching_courses(user_query: str, top_n: int = 10):
    query = normalize_query(user_query)

    partial_matches = course_kb[
        course_kb["course_name"].str.lower().apply(lambda x: query in normalize_query(x))
    ].copy()

    alias_hits = []
    for course, aliases in COURSE_ALIASES.items():
        normalized_aliases = [normalize_query(a) for a in aliases]
        if any(alias in query or query in alias for alias in normalized_aliases):
            alias_hits.append(course)

    alias_matches = course_kb[course_kb["course_name"].isin(alias_hits)].copy()

    matched = pd.concat([partial_matches, alias_matches], ignore_index=True).drop_duplicates(
        subset=["course_name", "instructor", "difficulty_level"]
    )

    if matched.empty:
        normalized_course_names = {course: normalize_query(course) for course in COURSE_NAMES}
        fuzzy_candidates = get_close_matches(
            query,
            list(normalized_course_names.values()),
            n=top_n,
            cutoff=0.35
        )

        if fuzzy_candidates:
            original_names = [
                original for original, norm in normalized_course_names.items()
                if norm in fuzzy_candidates
            ]
            matched = course_kb[course_kb["course_name"].isin(original_names)].copy()

    if not matched.empty:
        sort_cols = ["rating", "feedback_score"]
        if "enrollment_numbers" in matched.columns:
            sort_cols.append("enrollment_numbers")

        matched = matched.sort_values(
            by=sort_cols,
            ascending=[False] * len(sort_cols)
        )
        return matched.head(top_n)

    query_vec = tfidf_vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]

    fallback = course_kb.iloc[top_indices].copy()
    fallback["similarity_score"] = sim_scores[top_indices].round(4)
    return fallback

# =====================================================
# Endpoints
# =====================================================
@app.get("/")
def home():
    return {"message": "Online Course Recommendation API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "Enhanced Embedding (model2)",
        "version": "3.0.0"
    }

@app.get("/sample-users")
def sample_users():
    return {"sample_users": df["user_id"].drop_duplicates().head(20).tolist()}

@app.get("/new-users")
def get_new_users():
    return {
        "total_new_users": len(new_users_df),
        "new_users": new_users_df.to_dict(orient="records")
    }

@app.post("/suggest-users")
def suggest_users(request: UserSuggestRequest):
    prefix = request.prefix.strip()
    if not prefix:
        return {"suggestions": []}

    base_user_ids = df["user_id"].drop_duplicates()
    new_user_ids = new_users_df["user_id"].drop_duplicates() if not new_users_df.empty else pd.Series(dtype=int)
    user_ids = pd.concat([base_user_ids, new_user_ids], ignore_index=True).drop_duplicates().astype(str)
    matched = user_ids[user_ids.str.startswith(prefix)].head(20).tolist()
    return {"suggestions": matched}

@app.post("/suggest-courses")
def suggest_courses(request: CourseSuggestRequest):
    prefix = normalize_query(request.prefix)
    if not prefix:
        return {"suggestions": []}

    course_names = course_kb["course_name"].drop_duplicates()

    matched = course_names[
        course_names.str.lower().apply(lambda x: prefix in normalize_query(x))
    ].head(20).tolist()

    if not matched:
        normalized_course_names = {course: normalize_query(course) for course in COURSE_NAMES}
        fuzzy_candidates = get_close_matches(
            prefix,
            list(normalized_course_names.values()),
            n=10,
            cutoff=0.35
        )
        matched = [
            original for original, norm in normalized_course_names.items()
            if norm in fuzzy_candidates
        ]

    return {"suggestions": matched[:20]}

@app.post("/recommend")
def recommend_courses(request: RecommendRequest):
    user_id = request.user_id
    top_n = 5

    user_row = df[df["user_id"] == user_id]
    new_user_row = new_users_df[new_users_df["user_id"] == user_id]

    if user_row.empty and new_user_row.empty:
        return {
            "message": (
                f"No user found with user_id '{user_id}'. "
                f"If you are a new user, please go to New User Onboarding, "
                f"search/select a course, and create your new profile."
            ),
            "user_id": user_id,
            "user_history": [],
            "recommendations": []
        }

    if not user_row.empty:
        taken_courses = df[df["user_id"] == user_id][[
            "course_name", "instructor", "difficulty_level", "rating"
        ]].drop_duplicates().to_dict(orient="records")

        recommendations = recommend_top_courses_for_user(model, user_id, df, top_n=top_n)
    else:
        taken_courses = new_user_row[[
            "course_name", "instructor", "difficulty_level", "rating"
        ]].drop_duplicates().to_dict(orient="records")

        selected_course_name = str(new_user_row["course_name"].iloc[0])
        recommendations = recommend_related_from_selected_course(selected_course_name, top_n=top_n)

    return {
        "user_id": user_id,
        "user_history": taken_courses,
        "recommendations": recommendations
    }

@app.post("/search")
def search_courses(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    matched_courses_df = find_best_matching_courses(query, top_n=10)

    columns_to_return = [
        "course_name",
        "instructor",
        "difficulty_level",
        "certification_offered",
        "study_material_available",
        "course_price",
        "feedback_score",
        "rating"
    ]

    if "similarity_score" in matched_courses_df.columns:
        columns_to_return.append("similarity_score")

    results = matched_courses_df[columns_to_return].reset_index(drop=True).to_dict(orient="records")

    return {
        "query": query,
        "total_results": len(results),
        "results": results
    }

@app.post("/search-new-user-course")
def search_new_user_course(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    matched_courses_df = find_best_matching_courses(query, top_n=10)

    columns_to_return = [
        "course_name",
        "instructor",
        "difficulty_level",
        "rating",
        "feedback_score"
    ]

    if "similarity_score" in matched_courses_df.columns:
        columns_to_return.append("similarity_score")

    results = matched_courses_df[columns_to_return].reset_index(drop=True).to_dict(orient="records")

    return {
        "query": query,
        "matched_courses": results
    }

@app.post("/create-new-user")
def create_new_user(request: CreateNewUserRequest):
    global new_users_df

    course_name = request.course_name.strip()
    instructor = request.instructor.strip()
    difficulty_level = request.difficulty_level.strip()
    rating = float(request.rating)

    matched_df = df[
        (df["course_name"].str.lower() == course_name.lower()) &
        (df["instructor"].str.lower() == instructor.lower()) &
        (df["difficulty_level"].str.lower() == difficulty_level.lower())
    ]

    if matched_df.empty:
        raise HTTPException(
            status_code=404,
            detail="Selected course combination not found."
        )

    new_user_id = get_next_new_user_id()

    new_row = pd.DataFrame([{
        "user_id": new_user_id,
        "course_name": course_name,
        "instructor": instructor,
        "difficulty_level": difficulty_level,
        "rating": rating
    }])

    new_users_df = pd.concat([new_users_df, new_row], ignore_index=True)
    new_users_df.to_csv(NEW_USERS_PATH, index=False)

    related_courses = recommend_related_from_selected_course(course_name, top_n=5)

    return {
        "message": "New user created successfully.",
        "new_user_id": new_user_id,
        "selected_course_details": {
            "course_name": course_name,
            "instructor": instructor,
            "difficulty_level": difficulty_level,
            "rating": rating
        },
        "related_recommendations": related_courses
    }

@app.post("/chat")
def chat(request: ChatRequest):
    """
    LLM + RAG Chat endpoint.
    Flow:
      1. Retrieve top relevant courses using TF-IDF RAG
      2. Optionally load user history for personalization
      3. Build grounded context prompt
      4. Call Groq LLM (Llama 3) for natural language response
    """
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Please set the GROQ_API_KEY environment variable."
        )

    message = request.message.strip()
    user_id = request.user_id

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Step 1: Always retrieve fresh courses for the current query
    freshly_retrieved = retrieve_courses(message, top_n=5)

    # Step 2: Merge with ALL courses seen in this conversation (deduplicated by course_name+instructor)
    seen_keys = set()
    merged_courses = []
    for c in list(request.all_retrieved_courses) + freshly_retrieved:
        key = (c.get("course_name", ""), c.get("instructor", ""))
        if key not in seen_keys:
            seen_keys.add(key)
            merged_courses.append(c)

    retrieved = merged_courses

    # Step 2: Build course context from retrieved results
    course_context = ""
    for i, course in enumerate(retrieved, 1):
        course_context += (
            f"{i}. {course['course_name']}\n"
            f"   Instructor: {course['instructor']}\n"
            f"   Difficulty: {course['difficulty_level']}\n"
            f"   Certification: {course['certification_offered']}\n"
            f"   Study Material: {course['study_material_available']}\n"
            f"   Price: ${course['course_price']}\n"
            f"   Rating: {course['rating']} | Feedback Score: {course['feedback_score']}\n\n"
        )

    # Step 3: Optionally add user history for personalization
    # Check both main dataset and new users dataset
    user_context = ""
    if user_id is not None:
        user_row = df[df["user_id"] == user_id]
        new_user_row = new_users_df[new_users_df["user_id"] == user_id] if not new_users_df.empty else pd.DataFrame()

        if not user_row.empty:
            taken = (
                df[df["user_id"] == user_id][["course_name", "difficulty_level", "rating"]]
                .drop_duplicates()
            )
            user_context = f"User ID {user_id} is an existing user. Their past course history:\n"
            for _, row in taken.iterrows():
                user_context += f"  - {row['course_name']} ({row['difficulty_level']}, Rating given: {row['rating']})\n"
            user_context += "\n"

        elif not new_user_row.empty:
            user_context = f"User ID {user_id} is a new user. Their selected course:\n"
            for _, row in new_user_row.iterrows():
                user_context += f"  - {row['course_name']} ({row['difficulty_level']}, Rating given: {row['rating']})\n"
            user_context += "\n"

        else:
            user_context = f"User ID {user_id} was provided but not found in the database.\n"

    # Step 4: Build system prompt
    personalization_block = ""
    if user_context:
        personalization_block = f"""
The user's past course history has already been retrieved from the database and is provided below.
Use this history to personalize your recommendation — avoid recommending courses they have already taken.

{user_context}"""
    else:
        personalization_block = "\nNo user history available. Give a general recommendation based on the query.\n"

    system_prompt = f"""You are CourseGenie, a smart and friendly AI course recommendation assistant.
Your job is to help users find the best online courses based on their needs.

STRICT RULES:
- You already have all the data you need below — do NOT say you cannot access user data
- NEVER EVER deny or contradict your previous responses in the conversation history
- The conversation history above is the ground truth — always stay consistent with it
- If you previously listed multiple instructors, REMEMBER them and refer to them when asked
- If user asks "any other than X" — look at your previous response and suggest the OTHER instructors/courses you already listed
- If user asks a follow-up — always base your answer on the conversation history AND the retrieved courses below
- Only recommend courses from the Retrieved Courses Context below
- Do NOT make up or hallucinate any course names, instructors, or details
- Be specific, helpful, and conversational

Retrieved Courses Context (from database):
{course_context}{personalization_block}"""

    # Step 5: Build messages with full conversation history
    messages = [{"role": "system", "content": system_prompt}]

    # Add previous conversation turns so LLM remembers context
    if request.history:
        for turn in request.history:
            messages.append({"role": turn.role, "content": turn.content})

    # Add current user message
    messages.append({"role": "user", "content": message})

    # Call Groq LLM
    groq_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    llm_reply = groq_response.choices[0].message.content

    return {
        "message": message,
        "user_id": user_id,
        "retrieved_courses": retrieved,
        "response": llm_reply
    }


print("Swagger Docs: http://127.0.0.1:8001/docs")
print("Health Check: http://127.0.0.1:8001/health")
print("New Users: http://127.0.0.1:8001/new-users")