from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import os
import re

app = FastAPI(
    title="Online Course Recommendation API",
    description="Personalized course recommendations using Deep Learning + TF-IDF Retrieval",
    version="3.0.0"
)

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

# =====================================================
# Load Artifacts
# =====================================================
print("Loading deep learning model...")
model = tf.keras.models.load_model(MODEL_PATH)

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

    # if old schema exists, recreate with new schema
    if set(new_users_df.columns) != expected_columns:
        print("Old new_users.csv schema detected. Recreating with new schema...")
        new_users_df = pd.DataFrame(columns=expected_columns)
        new_users_df.to_csv(NEW_USERS_PATH, index=False)
    else:
        new_users_df = new_users_df[excepted_columns]
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

def retrieve_courses(query: str, top_n: int = 5):
    query = normalize_query(query)
    query_vec = tfidf_vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]

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

    user_ids = df["user_id"].drop_duplicates().astype(str)
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

    if user_row.empty:
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

    taken_courses = df[df["user_id"] == user_id][[
        "course_name", "instructor", "difficulty_level", "rating"
    ]].drop_duplicates().to_dict(orient="records")

    recommendations = recommend_top_courses_for_user(model, user_id, df, top_n=top_n)

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

print("Swagger Docs: http://127.0.0.1:8001/docs")
print("Health Check: http://127.0.0.1:8001/health")
print("New Users: http://127.0.0.1:8001/new-users")