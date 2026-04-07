# Online Course Recommendation System

A full-stack machine learning project that recommends online courses using deep learning, smart search, and new-user onboarding.

## Live Demo
https://online-course-recommendation-systems.streamlit.app/

## Overview

This project helps users discover relevant online courses through two main flows:

- **Existing users** get personalized recommendations based on their course history
- **New users** can search for a course, select one they like, create a profile, and receive related recommendations

The system combines:
- traditional recommender approaches
- deep learning–based personalized recommendation
- TF-IDF–based smart course retrieval
- a deployed interactive frontend

---

## Key Features

### Personalized Recommendations for Existing Users
Users can enter their user ID and get:
- previous course history
- personalized recommended courses

### Smart Course Search
Users can search with keywords such as:
- python
- machine
- ai
- cyber
- photo

The search layer supports:
- partial words
- abbreviations
- similar terms
- query normalization and matching

### New User Onboarding
For cold-start users:
- search for a course
- review matched course options
- select one exact course
- create a new user profile
- receive related course recommendations

### New User Storage
Newly created users are stored with:
- user ID
- course name
- instructor
- difficulty level
- rating

---

## Models Implemented

### Traditional Recommendation Systems
- Popularity-Based Recommender
- Content-Based Recommender
- Collaborative Filtering
- Hybrid Recommender

### Deep Learning Models
- Basic Embedding Model
- Enhanced Embedding Model
- Wide & Deep Model
- DeepFM Model

---

## Final Best Model

The final selected model is the **Enhanced Embedding Recommender**.

It achieved the best overall performance in the final comparison and is used for personalized recommendations for existing users.

---

## Search and Retrieval

A course knowledge base was created at the course level, and a **TF-IDF retriever** was built for:

- keyword-based course search
- smart matching for user queries
- cold-start support for new users

---

## Tech Stack

**Backend**
- FastAPI

**Frontend**
- Streamlit

**ML / DL**
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn

**Retrieval**
- TF-IDF
- Cosine Similarity

---

## Workflow

1. Data preprocessing and feature engineering  
2. Exploratory data analysis  
3. Traditional recommender systems  
4. Deep learning model comparison  
5. Best model selection  
6. Course knowledge base creation  
7. TF-IDF search and retrieval  
8. New user onboarding logic  
9. Backend and frontend integration  
10. Deployment  

---

## Project Structure

```bash
.
├── app/
│   ├── main.py
│   └── model/
│       ├── model2.keras
│       ├── user_encoder.pkl
│       ├── course_encoder.pkl
│       ├── tfidf_vectorizer.pkl
│       └── tfidf_matrix.pkl
├── data/
│   ├── processed_online_course_data.csv
│   ├── course_knowledge_base.csv
│   └── new_users.csv
├── requirements.txt
├── README.md
└── streamlit_app.py
