import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "").strip()


def beautify_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {
        "course_name": "Course Name",
        "instructor": "Instructor",
        "difficulty_level": "Difficulty Level",
        "predicted_rating": "Predicted Rating",
        "certification_offered": "Certification",
        "study_material_available": "Study Material",
        "course_price": "Course Price",
        "feedback_score": "Feedback Score",
        "rating": "Rating",
        "similarity_score": "Similarity Score",
        "selected_course": "Selected Course",
        "user_id": "User ID",
    }
    return df.rename(columns=rename_map)


def to_df(data: Any) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    return beautify_df(pd.DataFrame(data))


@st.cache_resource(show_spinner=False)
def get_backend() -> dict[str, Any]:
    if API_BASE_URL:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            return {
                "mode": "api",
                "message": (
                    f"Backend Connected via API | Status: {data.get('status')} | "
                    f"Model: {data.get('model')}"
                ),
            }
        except Exception as exc:
            return {
                "mode": "error",
                "message": f"Configured API backend is not reachable at {API_BASE_URL}. Error: {exc}",
            }

    try:
        from app import main as backend_module

        data = backend_module.health_check()
        return {
            "mode": "inprocess",
            "backend": backend_module,
            "message": (
                f"Backend Loaded In-Process | Status: {data.get('status')} | "
                f"Model: {data.get('model')}"
            ),
        }
    except Exception as exc:
        return {
            "mode": "error",
            "message": f"Unable to load backend module. Error: {exc}",
        }


def backend_health(backend_info: dict[str, Any]) -> tuple[bool, str]:
    if backend_info["mode"] in {"api", "inprocess"}:
        return True, backend_info["message"]
    return False, backend_info["message"]


def post_json(path: str, payload: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    backend_info = get_backend()

    if backend_info["mode"] == "api":
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    if backend_info["mode"] != "inprocess":
        raise RuntimeError(backend_info["message"])

    backend = backend_info["backend"]

    try:
        if path == "/suggest-users":
            return backend.suggest_users(backend.UserSuggestRequest(**payload))
        if path == "/suggest-courses":
            return backend.suggest_courses(backend.CourseSuggestRequest(**payload))
        if path == "/recommend":
            return backend.recommend_courses(backend.RecommendRequest(**payload))
        if path == "/search":
            return backend.search_courses(backend.SearchRequest(**payload))
        if path == "/search-new-user-course":
            return backend.search_new_user_course(backend.SearchRequest(**payload))
        if path == "/create-new-user":
            return backend.create_new_user(backend.CreateNewUserRequest(**payload))
    except Exception as exc:
        detail = getattr(exc, "detail", str(exc))
        raise RuntimeError(detail) from exc

    raise RuntimeError(f"Unsupported backend path: {path}")


def get_user_suggestions(prefix: str) -> list[str]:
    prefix = str(prefix).strip()
    if not prefix:
        return []
    try:
        data = post_json("/suggest-users", {"prefix": prefix}, timeout=10)
        return data.get("suggestions", [])
    except Exception:
        return []


def get_course_suggestions(prefix: str) -> list[str]:
    prefix = str(prefix).strip()
    if not prefix:
        return []
    try:
        data = post_json("/suggest-courses", {"prefix": prefix}, timeout=10)
        return data.get("suggestions", [])
    except Exception:
        return []


def build_exact_course_choices(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    choices = []
    for _, row in df.iterrows():
        course_name = row.get("Course Name", "")
        instructor = row.get("Instructor", "")
        difficulty = row.get("Difficulty Level", "")
        rating = row.get("Rating", "")
        choices.append(f"{course_name} | {instructor} | {difficulty} | Rating: {rating}")
    return choices


def init_state() -> None:
    defaults = {
        "existing_status": "",
        "existing_history": pd.DataFrame(),
        "existing_recommendations": pd.DataFrame(),
        "search_status": "",
        "search_results": pd.DataFrame(),
        "new_user_status": "",
        "new_user_matches": pd.DataFrame(),
        "new_user_exact_choices": [],
        "new_user_summary": "",
        "new_user_related": pd.DataFrame(),
        "new_user_created_id": None,
        "existing_user_input": "",
        "existing_user_select": None,
        "search_value": None,
        "new_user_value": None,
        "new_user_exact_selection": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_existing_user() -> None:
    st.session_state["existing_user_input"] = ""
    st.session_state["existing_user_select"] = None
    st.session_state["existing_status"] = ""
    st.session_state["existing_history"] = pd.DataFrame()
    st.session_state["existing_recommendations"] = pd.DataFrame()


def reset_search() -> None:
    st.session_state["search_value"] = None
    st.session_state["search_status"] = ""
    st.session_state["search_results"] = pd.DataFrame()


def reset_new_user() -> None:
    st.session_state["new_user_value"] = None
    st.session_state["new_user_status"] = ""
    st.session_state["new_user_matches"] = pd.DataFrame()
    st.session_state["new_user_exact_choices"] = []
    st.session_state["new_user_exact_selection"] = None
    st.session_state["new_user_summary"] = ""
    st.session_state["new_user_related"] = pd.DataFrame()


st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide",
)

init_state()
backend_info = get_backend()
healthy, health_message = backend_health(backend_info)

st.title("Online Course Recommendation System")
st.caption("Enhanced Embedding + Smart Search + New User Onboarding")

if healthy:
    st.success(health_message)
else:
    st.error(health_message)
    st.stop()

tab_existing, tab_search, tab_new_user = st.tabs(
    ["Existing User Recommendation", "Course Search", "New User Onboarding"]
)

with tab_existing:
    st.subheader("Existing User Flow")
    st.write("Enter a user ID directly, or choose from the matching suggestions.")

    col_input, col_select = st.columns([1, 1])

    with col_input:
        st.text_input("Enter User ID", key="existing_user_input")

    suggestions = get_user_suggestions(st.session_state["existing_user_input"])

    with col_select:
        if suggestions:
            st.selectbox(
                "Matching User IDs",
                options=suggestions,
                index=None,
                placeholder="Select a suggested user ID",
                key="existing_user_select",
            )
        else:
            st.session_state["existing_user_select"] = None
            st.caption("No user suggestions yet.")

    col_action, col_reset = st.columns([1, 1])
    with col_action:
        if st.button("Get Recommendations", use_container_width=True):
            user_value = st.session_state["existing_user_select"] or st.session_state["existing_user_input"]
            if not str(user_value).strip():
                st.session_state["existing_status"] = "Please enter a valid user ID."
                st.session_state["existing_history"] = pd.DataFrame()
                st.session_state["existing_recommendations"] = pd.DataFrame()
            else:
                try:
                    data = post_json("/recommend", {"user_id": int(user_value)})
                    st.session_state["existing_history"] = to_df(data.get("user_history", []))
                    st.session_state["existing_recommendations"] = to_df(data.get("recommendations", []))
                    if data.get("user_history"):
                        st.session_state["existing_status"] = (
                            f"Showing previous courses and recommendations for User ID: {user_value}"
                        )
                    else:
                        st.session_state["existing_status"] = data.get(
                            "message",
                            f"No history found for User ID: {user_value}",
                        )
                except Exception as exc:
                    st.session_state["existing_status"] = f"Error: {exc}"
                    st.session_state["existing_history"] = pd.DataFrame()
                    st.session_state["existing_recommendations"] = pd.DataFrame()

    with col_reset:
        if st.button("Reset Existing User", use_container_width=True):
            reset_existing_user()
            st.rerun()

    if st.session_state["existing_status"]:
        st.info(st.session_state["existing_status"])

    st.dataframe(st.session_state["existing_history"], use_container_width=True)
    st.dataframe(st.session_state["existing_recommendations"], use_container_width=True)

with tab_search:
    st.subheader("Course Search")
    st.write("Type inside the dropdown to see course suggestions in the same field.")

    current_search_value = st.session_state["search_value"] or ""
    search_suggestions = get_course_suggestions(current_search_value)

    st.selectbox(
        "Enter Course / Keyword",
        options=search_suggestions,
        index=None,
        placeholder="Type a course keyword here",
        accept_new_options=True,
        key="search_value",
    )

    col_action, col_reset = st.columns([1, 1])
    with col_action:
        if st.button("Search Courses", use_container_width=True):
            course_value = st.session_state["search_value"]
            if not str(course_value).strip():
                st.session_state["search_status"] = "Please enter a course keyword."
                st.session_state["search_results"] = pd.DataFrame()
            else:
                current_suggestions = get_course_suggestions(str(course_value))
                if not current_suggestions:
                    st.session_state["search_status"] = (
                        f"No matching course suggestions found for: {course_value}"
                    )
                    st.session_state["search_results"] = pd.DataFrame()
                else:
                    try:
                        data = post_json("/search", {"query": course_value})
                        results = to_df(data.get("results", []))
                        st.session_state["search_status"] = (
                            f"Found {len(results)} matching courses for: {course_value}"
                        )
                        st.session_state["search_results"] = results
                    except Exception as exc:
                        st.session_state["search_status"] = f"Error: {exc}"
                        st.session_state["search_results"] = pd.DataFrame()

    with col_reset:
        if st.button("Reset Course Search", use_container_width=True):
            reset_search()
            st.rerun()

    if st.session_state["search_status"]:
        st.info(st.session_state["search_status"])

    st.dataframe(st.session_state["search_results"], use_container_width=True)

with tab_new_user:
    st.subheader("New User Flow")
    st.write(
        "Type inside the dropdown to see course suggestions in the same field, then choose one exact course."
    )

    current_new_user_value = st.session_state["new_user_value"] or ""
    new_user_suggestions = get_course_suggestions(current_new_user_value)

    st.selectbox(
        "Enter Course / Keyword for New User",
        options=new_user_suggestions,
        index=None,
        placeholder="Type a course keyword here",
        accept_new_options=True,
        key="new_user_value",
    )

    col_search, col_reset = st.columns([1, 1])
    with col_search:
        if st.button("Search Course For New User", use_container_width=True):
            course_value = st.session_state["new_user_value"]
            if not str(course_value).strip():
                st.session_state["new_user_status"] = "Please enter a course keyword."
                st.session_state["new_user_matches"] = pd.DataFrame()
                st.session_state["new_user_exact_choices"] = []
                st.session_state["new_user_exact_selection"] = None
            else:
                current_suggestions = get_course_suggestions(str(course_value))
                if not current_suggestions:
                    st.session_state["new_user_status"] = (
                        f"No matching course suggestions found for: {course_value}"
                    )
                    st.session_state["new_user_matches"] = pd.DataFrame()
                    st.session_state["new_user_exact_choices"] = []
                    st.session_state["new_user_exact_selection"] = None
                else:
                    try:
                        data = post_json("/search-new-user-course", {"query": course_value})
                        matched_df = to_df(data.get("matched_courses", []))
                        st.session_state["new_user_matches"] = matched_df
                        st.session_state["new_user_exact_choices"] = build_exact_course_choices(matched_df)
                        st.session_state["new_user_exact_selection"] = None
                        st.session_state["new_user_status"] = (
                            f"Found {len(matched_df)} matched courses for: {course_value}. "
                            "Please select one exact course below."
                        )
                    except Exception as exc:
                        st.session_state["new_user_status"] = f"Error: {exc}"
                        st.session_state["new_user_matches"] = pd.DataFrame()
                        st.session_state["new_user_exact_choices"] = []
                        st.session_state["new_user_exact_selection"] = None

    with col_reset:
        if st.button("Reset New User Flow", use_container_width=True):
            reset_new_user()
            st.rerun()

    if st.session_state["new_user_status"]:
        st.info(st.session_state["new_user_status"])

    st.dataframe(st.session_state["new_user_matches"], use_container_width=True)

    exact_choices = st.session_state["new_user_exact_choices"]
    if exact_choices:
        st.selectbox(
            "Select Exact Course",
            options=exact_choices,
            index=None,
            placeholder="Choose one exact course",
            key="new_user_exact_selection",
        )

    if st.button("Create New User", use_container_width=True):
        selected_exact_course = st.session_state["new_user_exact_selection"]
        if not selected_exact_course or not str(selected_exact_course).strip():
            st.session_state["new_user_summary"] = "Please select one exact course before creating the user."
            st.session_state["new_user_related"] = pd.DataFrame()
        else:
            try:
                parts = [part.strip() for part in selected_exact_course.split("|")]
                if len(parts) < 4:
                    raise ValueError("Invalid course selection format.")

                course_name = parts[0]
                instructor = parts[1]
                difficulty_level = parts[2]
                rating = float(parts[3].replace("Rating:", "").strip())

                payload = {
                    "course_name": course_name,
                    "instructor": instructor,
                    "difficulty_level": difficulty_level,
                    "rating": rating,
                }

                data = post_json("/create-new-user", payload)
                selected = data.get("selected_course_details", {})
                new_user_id = data.get("new_user_id")

                st.session_state["new_user_created_id"] = new_user_id
                st.session_state["existing_user_input"] = str(new_user_id)
                st.session_state["existing_user_select"] = str(new_user_id)
                st.session_state["new_user_summary"] = (
                    f"New User ID: {new_user_id}\n"
                    f"Course Name: {selected.get('course_name')}\n"
                    f"Instructor: {selected.get('instructor')}\n"
                    f"Difficulty Level: {selected.get('difficulty_level')}\n"
                    f"Rating: {selected.get('rating')}\n"
                    f"{data.get('message', '')}"
                )
                st.session_state["new_user_related"] = to_df(data.get("related_recommendations", []))
            except Exception as exc:
                st.session_state["new_user_summary"] = f"Error: {exc}"
                st.session_state["new_user_related"] = pd.DataFrame()

    if st.session_state["new_user_summary"]:
        st.text_area("New User Summary", st.session_state["new_user_summary"], height=160)

    if st.session_state["new_user_created_id"] is not None:
        st.success(
            f"New user ID {st.session_state['new_user_created_id']} is ready. "
            "You can now open the Existing User Recommendation tab and fetch it directly."
        )

    st.dataframe(st.session_state["new_user_related"], use_container_width=True)
