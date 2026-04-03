import gradio as gr
import requests
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8001"


# =====================================================
# Helpers
# =====================================================
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


def to_df(data):
    if not data:
        return pd.DataFrame()
    return beautify_df(pd.DataFrame(data))


def empty_df():
    return pd.DataFrame()


# =====================================================
# Backend health
# =====================================================
def check_backend():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        return f"Backend Connected | Status: {data.get('status')} | Model: {data.get('model')}"
    except Exception as e:
        return f"Backend Not Reachable | Error: {str(e)}"


# =====================================================
# Suggestion loaders
# =====================================================
def update_user_dropdown(text):
    prefix = str(text).strip()
    if not prefix:
        return gr.update(choices=[], value=None)

    try:
        response = requests.post(
            f"{API_BASE_URL}/suggest-users",
            json={"prefix": prefix},
            timeout=10
        )
        response.raise_for_status()
        suggestions = response.json().get("suggestions", [])
        return gr.update(choices=suggestions, value=prefix)
    except Exception:
        return gr.update(choices=[], value=prefix)


def update_course_dropdown(text):
    prefix = str(text).strip()
    if not prefix:
        return gr.update(choices=[], value=None)

    try:
        response = requests.post(
            f"{API_BASE_URL}/suggest-courses",
            json={"prefix": prefix},
            timeout=10
        )
        response.raise_for_status()
        suggestions = response.json().get("suggestions", [])
        return gr.update(choices=suggestions, value=prefix)
    except Exception:
        return gr.update(choices=[], value=prefix)


# =====================================================
# Existing User
# =====================================================
def get_existing_user_recommendations(user_value):
    if user_value is None or str(user_value).strip() == "":
        return (
            "Please enter a valid user ID.",
            empty_df(),
            empty_df()
        )

    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={"user_id": int(user_value)},
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        user_history = to_df(data.get("user_history", []))
        recommendations = to_df(data.get("recommendations", []))

        if data.get("user_history"):
            status_msg = f"Showing previous courses and recommendations for User ID: {user_value}"
        else:
            status_msg = data.get(
                "message",
                f"No history found for User ID: {user_value}"
            )

        return status_msg, user_history, recommendations

    except Exception as e:
        return f"Error: {str(e)}", empty_df(), empty_df()


def reset_existing_user():
    return None, "", empty_df(), empty_df()


# =====================================================
# Course Search
# =====================================================
def search_courses(course_value):
    if course_value is None or str(course_value).strip() == "":
        return "Please enter a course keyword.", empty_df()

    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": course_value},
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        results = to_df(data.get("results", []))
        return f"Found {len(results)} matching courses for: {course_value}", results

    except Exception as e:
        return f"Error: {str(e)}", empty_df()


def reset_search():
    return None, "", empty_df()


# =====================================================
# New User Onboarding
# =====================================================
def build_exact_course_choices(df):
    if df is None or df.empty:
        return []

    choices = []
    for _, row in df.iterrows():
        course_name = row.get("Course Name", "")
        instructor = row.get("Instructor", "")
        difficulty = row.get("Difficulty Level", "")
        rating = row.get("Rating", "")

        label = f"{course_name} | {instructor} | {difficulty} | Rating: {rating}"
        choices.append(label)

    return choices


def search_new_user_courses(course_value):
    if course_value is None or str(course_value).strip() == "":
        return (
            "Please enter a course keyword.",
            empty_df(),
            gr.update(choices=[], value=None)
        )

    try:
        response = requests.post(
            f"{API_BASE_URL}/search-new-user-course",
            json={"query": course_value},
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        matched_courses = data.get("matched_courses", [])
        matched_df = to_df(matched_courses)

        exact_choices = build_exact_course_choices(matched_df)

        return (
            f"Found {len(matched_df)} matched courses for: {course_value}. Please select one exact course below.",
            matched_df,
            gr.update(choices=exact_choices, value=None)
        )

    except Exception as e:
        return (
            f"Error: {str(e)}",
            empty_df(),
            gr.update(choices=[], value=None)
        )


def create_new_user(selected_exact_course):
    if not selected_exact_course or not str(selected_exact_course).strip():
        return (
            "Please select one exact course before creating the user.",
            empty_df(),
            None,
            "",
            empty_df(),
            gr.update(choices=[], value=None)
        )

    try:
        parts = [p.strip() for p in selected_exact_course.split("|")]
        if len(parts) < 4:
            return (
                "Invalid course selection format.",
                empty_df(),
                None,
                "",
                empty_df(),
                gr.update(choices=[], value=None)
            )

        course_name = parts[0]
        instructor = parts[1]
        difficulty_level = parts[2]
        rating_text = parts[3].replace("Rating:", "").strip()
        rating = float(rating_text)

        payload = {
            "course_name": course_name,
            "instructor": instructor,
            "difficulty_level": difficulty_level,
            "rating": rating
        }

        response = requests.post(
            f"{API_BASE_URL}/create-new-user",
            json=payload,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

        selected = data.get("selected_course_details", {})

        summary = (
            f"New User ID: {data.get('new_user_id')}\n"
            f"Course Name: {selected.get('course_name')}\n"
            f"Instructor: {selected.get('instructor')}\n"
            f"Difficulty Level: {selected.get('difficulty_level')}\n"
            f"Rating: {selected.get('rating')}\n"
            f"{data.get('message', '')}"
        )

        related_df = to_df(data.get("related_recommendations", []))

        return (
            summary,
            related_df,
            None,                              # reset main input
            "",                                # reset search status
            empty_df(),                        # reset matched table
            gr.update(choices=[], value=None)  # reset exact selection
        )

    except Exception as e:
        return (
            f"Error: {str(e)}",
            empty_df(),
            None,
            "",
            empty_df(),
            gr.update(choices=[], value=None)
        )


def reset_new_user():
    return None, "", empty_df(), gr.update(choices=[], value=None), "", empty_df()


# =====================================================
# UI
# =====================================================
custom_css = """
.gradio-container {
    width: 96vw !important;
    max-width: 96vw !important;
    margin: 0 auto !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
}
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 10px;
}
.sub-title {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 24px;
}
.section-note {
    font-size: 16px;
    color: #444;
    margin-bottom: 10px;
}
.gr-button {
    min-height: 46px !important;
}
.gr-dataframe {
    min-height: 260px !important;
}
"""

with gr.Blocks(title="Online Course Recommendation System", css=custom_css) as demo:
    gr.Markdown("<div class='main-title'>Online Course Recommendation System</div>")
    gr.Markdown("<div class='sub-title'>Enhanced Embedding + Smart Search + New User Onboarding</div>")

    backend_status = gr.Textbox(
        label="Backend Status",
        value=check_backend(),
        interactive=False
    )

    with gr.Tabs():

        # Existing User
        with gr.Tab("Existing User Recommendation"):
            gr.Markdown("### Existing User Flow")
            gr.Markdown(
                "<div class='section-note'>Enter your user ID directly in the field below. Suggestions appear in the same field flow.</div>"
            )

            existing_user_field = gr.Dropdown(
                label="Enter User ID",
                choices=[],
                allow_custom_value=True,
                interactive=True
            )

            existing_user_field.input(
                fn=update_user_dropdown,
                inputs=existing_user_field,
                outputs=existing_user_field
            )

            with gr.Row():
                existing_user_button = gr.Button("Get Recommendations", variant="primary")
                existing_user_reset = gr.Button("Reset")

            existing_user_status = gr.Textbox(label="Status", interactive=False)
            existing_user_history = gr.Dataframe(label="Your Previous Courses", interactive=False, wrap=True)
            existing_user_recommendations = gr.Dataframe(label="Recommended Courses", interactive=False, wrap=True)

            existing_user_button.click(
                fn=get_existing_user_recommendations,
                inputs=existing_user_field,
                outputs=[
                    existing_user_status,
                    existing_user_history,
                    existing_user_recommendations
                ]
            )

            existing_user_reset.click(
                fn=reset_existing_user,
                outputs=[
                    existing_user_field,
                    existing_user_status,
                    existing_user_history,
                    existing_user_recommendations
                ]
            )

        # Course Search
        with gr.Tab("Course Search"):
            gr.Markdown("### Course Search")
            gr.Markdown(
                "<div class='section-note'>Enter a course keyword directly in the field below. Suggestions appear in the same field flow.</div>"
            )

            search_field = gr.Dropdown(
                label="Enter Course / Keyword",
                choices=[],
                allow_custom_value=True,
                interactive=True
            )

            search_field.input(
                fn=update_course_dropdown,
                inputs=search_field,
                outputs=search_field
            )

            with gr.Row():
                search_button = gr.Button("Search Courses", variant="primary")
                search_reset = gr.Button("Reset")

            search_status = gr.Textbox(label="Status", interactive=False)
            search_results = gr.Dataframe(label="Search Results", interactive=False, wrap=True)

            search_button.click(
                fn=search_courses,
                inputs=search_field,
                outputs=[search_status, search_results]
            )

            search_reset.click(
                fn=reset_search,
                outputs=[search_field, search_status, search_results]
            )

        # New User Onboarding
        with gr.Tab("New User Onboarding"):
            gr.Markdown("### New User Flow")
            gr.Markdown(
                "<div class='section-note'>Enter a course keyword in the field below. Suggestions appear in the same field flow. Search first, review matched rows, then choose one exact course.</div>"
            )

            new_user_field = gr.Dropdown(
                label="Enter Course / Keyword",
                choices=[],
                allow_custom_value=True,
                interactive=True
            )

            new_user_field.input(
                fn=update_course_dropdown,
                inputs=new_user_field,
                outputs=new_user_field
            )

            with gr.Row():
                search_new_user_button = gr.Button("Search Course", variant="primary")
                new_user_reset = gr.Button("Reset")

            new_user_search_status = gr.Textbox(label="Search Status", interactive=False)
            new_user_matched_courses = gr.Dataframe(label="Matched Courses", interactive=False, wrap=True)

            final_selected_course = gr.Dropdown(
                label="Select Exact Course",
                choices=[],
                interactive=True
            )

            create_new_user_button = gr.Button("Create New User", variant="primary")

            new_user_summary = gr.Textbox(label="New User Summary", interactive=False)
            new_user_related_courses = gr.Dataframe(
                label="Related Recommendations",
                interactive=False,
                wrap=True
            )

            search_new_user_button.click(
                fn=search_new_user_courses,
                inputs=new_user_field,
                outputs=[
                    new_user_search_status,
                    new_user_matched_courses,
                    final_selected_course
                ]
            )

            create_new_user_button.click(
                fn=create_new_user,
                inputs=final_selected_course,
                outputs=[
                    new_user_summary,
                    new_user_related_courses,
                    new_user_field,
                    new_user_search_status,
                    new_user_matched_courses,
                    final_selected_course
                ]
            )

            new_user_reset.click(
                fn=reset_new_user,
                outputs=[
                    new_user_field,
                    new_user_search_status,
                    new_user_matched_courses,
                    final_selected_course,
                    new_user_summary,
                    new_user_related_courses
                ]
            )

demo.launch()