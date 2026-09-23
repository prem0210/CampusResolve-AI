from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

LOCATION_TYPES = [
    "Hostel",
    "Classroom",
    "Laboratory",
    "Library",
    "Washroom",
    "Canteen",
    "Campus Building",
    "Campus Road",
    "Parking",
    "Office",
]

LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
    "Tamil-English / Tanglish": "ta_en",
}

STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]

PAGES = {
    "Report an issue": "report",
    "Track complaint": "track",
    "Staff workspace": "staff",
    "About": "about",
}

EXAMPLES = {
    "Water leakage": {
        "complaint_text": (
            "Hostel Block B la water leak aagudhu and floor slippery ah iruku."
        ),
        "language": "Tamil-English / Tanglish",
        "location_type": "Hostel",
        "specific_location": "Hostel Block B",
        "affected_population": 160,
        "safety_flag": True,
        "repeat_count": 2,
    },
    "Wi-Fi unavailable": {
        "complaint_text": (
            "The Wi-Fi in the central library reading hall is unavailable and "
            "students cannot access online learning materials."
        ),
        "language": "English",
        "location_type": "Library",
        "specific_location": "Central Library Reading Hall",
        "affected_population": 120,
        "safety_flag": False,
        "repeat_count": 1,
    },
    "Classroom facility": {
        "complaint_text": (
            "சி-204 வகுப்பறையில் மின்விசிறி வேலை செய்யவில்லை. "
            "வகுப்பு நடத்த சிரமமாக உள்ளது."
        ),
        "language": "Tamil",
        "location_type": "Classroom",
        "specific_location": "C-204",
        "affected_population": 45,
        "safety_flag": False,
        "repeat_count": 0,
    },
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #0F172A;
                --body: #475569;
                --muted: #64748B;
                --line: #E2E8F0;
                --surface: #F8FAFC;
                --blue: #2563EB;
                --blue-dark: #1D4ED8;
                --blue-soft: #EFF6FF;
                --green: #15803D;
                --amber: #B45309;
                --red: #B91C1C;
            }

            .stApp {
                background: #EFF6FF;
            }

            [data-testid="stHeader"] {
                background: rgba(244, 247, 251, 0.96);
                border-bottom: 1px solid rgba(226, 232, 240, 0.75);
            }

            .block-container {
                max-width: 1280px;
                padding: 1.5rem 2.25rem 3rem;
            }

            h1, h2, h3, h4 {
                color: var(--ink) !important;
                letter-spacing: -0.02em;
            }

            h1 {
                font-size: 2rem !important;
                font-weight: 800 !important;
                margin-bottom: 0.3rem !important;
            }

            h2 {
                font-size: 1.45rem !important;
                font-weight: 750 !important;
            }

            h3 {
                font-weight: 700 !important;
            }

            p, label, .stCaption {
                color: var(--body);
            }

            [data-testid="stSidebar"] {
                background: #F8FAFC;
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] .block-container {
                padding: 1.25rem 1rem 2rem;
            }

            [data-testid="stSidebar"] h2 {
                margin-top: 0 !important;
                font-size: 1.16rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] label {
                padding: 0.2rem 0;
                color: #334155;
                font-weight: 600;
            }

            .stButton > button,
            .stFormSubmitButton > button {
                width: 100%;
                min-height: 2.65rem;
                border: 1px solid var(--blue);
                border-radius: 8px;
                background: var(--blue);
                color: #FFFFFF;
                font-weight: 700;
                transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                background: var(--blue-dark);
                border-color: var(--blue-dark);
                color: #FFFFFF;
                box-shadow: 0 5px 14px rgba(37, 99, 235, 0.18);
            }

            .stButton > button:focus,
            .stFormSubmitButton > button:focus {
                outline: 3px solid rgba(37, 99, 235, 0.20);
                outline-offset: 2px;
            }

            .stTextInput input,
            .stTextArea textarea,
            .stNumberInput input,
            .stDateInput input,
            .stSelectbox div[data-baseweb="select"] > div {
                background: #FFFFFF !important;
                border-color: #CBD5E1 !important;
                border-radius: 8px !important;
            }

            .stTextInput input:focus,
            .stTextArea textarea:focus,
            .stNumberInput input:focus {
                border-color: var(--blue) !important;
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
            }

            [data-testid="stMetric"] {
                min-height: 118px;
                padding: 1rem;
                border: 1px solid var(--line);
                border-radius: 12px;
                background: #FFFFFF;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
            }

            [data-testid="stMetricLabel"] {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
            }

            [data-testid="stMetricValue"] {
                color: var(--ink);
                font-weight: 800;
            }

            [data-testid="stExpander"],
            [data-testid="stDataFrame"] {
                border: 1px solid var(--line);
                border-radius: 12px;
                overflow: hidden;
                background: #FFFFFF;
            }

            [data-testid="stAlert"] {
                border-radius: 10px;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-color: var(--line) !important;
                border-radius: 12px;
                background: #FFFFFF;
            }

            .brand-block {
                margin-bottom: 1.15rem;
            }

            .brand-name {
                color: var(--ink);
                font-size: 1.2rem;
                font-weight: 800;
                letter-spacing: -0.035em;
            }

            .brand-caption {
                margin-top: 0.16rem;
                color: var(--muted);
                font-size: 0.78rem;
            }

            .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin: 0 0 1.45rem;
                padding: 0 0 1rem;
                border-bottom: 1px solid var(--line);
            }

            .topbar-title {
                color: var(--ink);
                font-size: 1.65rem;
                font-weight: 800;
                letter-spacing: -0.035em;
            }

            .eyebrow {
                margin-bottom: 0.38rem;
                color: var(--blue);
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }

            .topbar-status {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                color: var(--muted);
                font-size: 0.84rem;
                font-weight: 600;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #16A34A;
                box-shadow: 0 0 0 4px #DCFCE7;
            }

            .page-heading {
                margin: 0.15rem 0 1rem;
            }

            .page-heading h2 {
                margin: 0 !important;
            }

            .page-heading p {
                max-width: 770px;
                margin: 0.4rem 0 0;
                color: var(--body);
                font-size: 0.95rem;
                line-height: 1.55;
            }

            .info-card {
                margin: 1.1rem 0 1.15rem;
                padding: 1.15rem 1.2rem;
                border: 1px solid var(--line);
                border-left: 4px solid var(--blue);
                border-radius: 10px;
                background: var(--surface);
            }

            .info-card-title {
                color: var(--ink);
                font-size: 0.96rem;
                font-weight: 800;
            }

            .info-card-body {
                margin-top: 0.32rem;
                color: var(--body);
                font-size: 0.91rem;
                line-height: 1.55;
            }

            .reference-card {
                margin: 0.8rem 0 1.25rem;
                padding: 1.15rem 1.25rem;
                border: 1px solid #BFDBFE;
                border-radius: 12px;
                background: #F8FBFF;
            }

            .reference-label {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .reference-value {
                margin-top: 0.28rem;
                color: var(--ink);
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 1.12rem;
                font-weight: 800;
            }

            .reference-detail {
                margin-top: 0.42rem;
                color: var(--body);
                font-size: 0.9rem;
            }

            .status-badge {
                display: inline-block;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 800;
                line-height: 1.2;
            }

            .footer {
                margin-top: 3rem;
                padding: 1.15rem 0 0;
                border-top: 1px solid var(--line);
                color: #94A3B8;
                font-size: 0.76rem;
                text-align: center;
            }

            @media (max-width: 760px) {
                .block-container {
                    padding: 1.1rem 1rem 2rem;
                }

                .topbar {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .topbar-title {
                    font-size: 1.45rem;
                }

                .topbar-status {
                    font-size: 0.78rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_auth_headers() -> dict[str, str]:
    token = str(st.session_state.get("access_token") or "").strip()

    if not token:
        return {}

    return {"Authorization": f"Bearer {token}"}


def api_get(
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}{path}",
        params=params,
        headers=get_auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        headers=get_auth_headers(),
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def api_patch(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.patch(
        f"{API_BASE_URL}{path}",
        json=payload,
        headers=get_auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def is_authenticated() -> bool:
    token = str(st.session_state.get("access_token") or "").strip()
    current_user = st.session_state.get("current_user")

    return bool(token and isinstance(current_user, dict))


def get_current_user_role() -> str:
    current_user = st.session_state.get("current_user")

    if not isinstance(current_user, dict):
        return ""

    return str(current_user.get("role") or "").strip()


def is_staff_or_admin() -> bool:
    return get_current_user_role() in {"Staff", "Admin"}


def login_user(email: str, password: str) -> tuple[bool, str]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={
                "email": email.strip(),
                "password": password,
            },
            timeout=30,
        )
    except requests.RequestException:
        return False, "Could not reach the authentication service. Please try again."

    if response.status_code == 401:
        return False, "Invalid email or password."

    if response.status_code == 422:
        return False, "Enter a valid email address and a password of at least 8 characters."

    try:
        response.raise_for_status()
        token_data = response.json()
    except requests.RequestException:
        return False, "Sign-in could not be completed. Please try again."

    access_token = str(token_data.get("access_token") or "").strip()

    if not access_token:
        return False, "Sign-in failed because the server did not return an access token."

    previous_token = st.session_state.get("access_token")
    previous_user = st.session_state.get("current_user")

    st.session_state["access_token"] = access_token

    try:
        current_user = api_get("/auth/me")
    except requests.RequestException:
        st.session_state["access_token"] = previous_token
        st.session_state["current_user"] = previous_user
        return False, "Sign-in could not be verified. Please try again."

    st.session_state["current_user"] = current_user
    st.session_state["auth_error"] = None

    return True, ""


def logout_user() -> None:
    st.session_state["access_token"] = None
    st.session_state["current_user"] = None
    st.session_state["auth_error"] = None
    st.session_state["active_page"] = "Report an issue"


def render_auth_panel() -> None:
    if is_authenticated():
        current_user = st.session_state["current_user"]
        full_name = str(current_user.get("full_name") or "Authenticated user")
        email = str(current_user.get("email") or "")
        role = get_current_user_role() or "User"

        st.success(f"Signed in as {full_name}")
        st.caption(f"{email} ? {role}")

        if st.button("Log out", use_container_width=True, key="logout_button"):
            logout_user()
            st.rerun()

        return

    st.markdown("### Staff and Admin sign in")
    st.caption(
        "Sign in to access protected complaint operations. "
        "Students can still submit and track complaints without signing in."
    )

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input(
            "Email",
            key="login_email",
            placeholder="name@campusresolve.example.com",
        )
        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )
        submitted = st.form_submit_button(
            "Sign in",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    if not email.strip() or not password:
        st.error("Enter both your email address and password.")
        return

    with st.spinner("Signing in..."):
        success, message = login_user(email, password)

    if not success:
        st.session_state["auth_error"] = message
        st.error(message)
        return

    st.session_state["auth_error"] = None
    st.rerun()


def get_api_health() -> tuple[bool, str]:
    try:
        data = api_get("/health")
        return bool(data.get("models_loaded")), str(data.get("status", "unknown"))
    except requests.RequestException:
        return False, "unreachable"


def priority_emoji(priority: str) -> str:
    return {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🟠",
        "Critical": "🔴",
    }.get(priority, "⚪")


def format_datetime(value: Any) -> str:
    if not value:
        return "Not available"
    try:
        return pd.to_datetime(value).strftime("%d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return str(value)


def status_badge(status: str) -> str:
    styles = {
        "Open": ("#EFF6FF", "#1D4ED8"),
        "In Progress": ("#FFFBEB", "#B45309"),
        "Resolved": ("#F0FDF4", "#15803D"),
        "Closed": ("#F1F5F9", "#475569"),
    }
    background, color = styles.get(status, ("#F1F5F9", "#475569"))
    return (
        f'<span class="status-badge" style="background:{background}; color:{color};">'
        f"{status}</span>"
    )


def render_page_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="page-heading">
            <div class="eyebrow">{kicker}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_example(example: dict[str, Any]) -> None:
    for key, value in example.items():
        st.session_state[key] = value
    st.session_state["last_submission"] = None


def open_tracker(reference: str) -> None:
    st.session_state["pending_track_reference"] = reference
    st.session_state["tracked_complaint"] = None
    st.session_state["active_page"] = "Track complaint"


def reset_complaint_form() -> None:
    st.session_state["last_submission"] = None
    st.session_state["complaint_text"] = ""
    st.session_state["specific_location"] = ""
    st.session_state["affected_population"] = 1
    st.session_state["safety_flag"] = False
    st.session_state["repeat_count"] = 0


def show_student_portal(api_ready: bool) -> None:
    render_page_heading(
        "Student support portal",
        "Report a campus issue",
        "Submit your concern in English, Tamil, or Tanglish. The system will recommend the appropriate department and urgency level for staff review.",
    )

    if not api_ready:
        st.error("The complaint service is unavailable. Please start the FastAPI backend and try again.")

    if not is_authenticated():
        st.info(
            "Please sign in to submit a complaint. "
            "Signing in links the complaint to your account for secure tracking."
        )

    with st.container(border=True):
        st.markdown("### Tell us what happened")
        st.caption("Share enough detail for the system to recommend the correct department and urgency level.")

        with st.form("complaint_form", clear_on_submit=False):
            complaint_text = st.text_area(
                "Complaint details",
                placeholder="Example: Hostel Block B la water leak aagudhu and floor slippery ah iruku.",
                height=155,
                key="complaint_text",
                help="Do not include passwords, IDs, phone numbers, or other sensitive personal information.",
            )

            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                language = st.selectbox("Language", list(LANGUAGES.keys()), key="language")
                location_type = st.selectbox("Location type", LOCATION_TYPES, key="location_type")
            with col2:
                specific_location = st.text_input(
                    "Specific location",
                    max_chars=150,
                    key="specific_location",
                    placeholder="Example: Hostel Block B, second floor",
                )
                affected_population = st.number_input(
                    "People affected",
                    min_value=1,
                    max_value=10000,
                    step=1,
                    key="affected_population",
                )
            with col3:
                safety_flag = st.checkbox("Safety or health concern", key="safety_flag")
                repeat_count = st.number_input(
                    "Earlier reports",
                    min_value=0,
                    max_value=100,
                    step=1,
                    key="repeat_count",
                )

            submitted = st.form_submit_button(
                "Submit complaint and get AI recommendation",
                type="primary",
                use_container_width=True,
                disabled=not api_ready or not is_authenticated(),
            )

    if submitted:
        if len(complaint_text.strip()) < 5:
            st.error("Please enter a complaint with at least 5 characters.")
            return

        payload = {
            "complaint_text": complaint_text.strip(),
            "language": LANGUAGES[language],
            "location_type": location_type,
            "specific_location": specific_location.strip() or "Not specified",
            "affected_population": int(affected_population),
            "safety_flag": int(safety_flag),
            "repeat_count": int(repeat_count),
        }
        try:
            with st.spinner("Submitting and analyzing your complaint..."):
                st.session_state["last_submission"] = api_post("/complaints", payload)
            st.success("Your complaint was submitted successfully.")
        except requests.RequestException as error:
            st.error("Submission failed. Please check that the backend is running and try again.")
            with st.expander("Technical details"):
                st.code(str(error))

    result = st.session_state.get("last_submission")
    if not result:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">What happens after submission?</div>
                <div class="info-card-body">
                    CampusResolve-AI recommends an issue category and department, checks for similar historic reports,
                    forecasts an urgency level, and estimates a likely resolution time. Campus staff review all
                    recommendations before taking operational action.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    reference = result["complaint_reference"]
    priority = result.get("predicted_priority", "Not available")
    duplicate_flag = bool(result.get("possible_duplicate"))

    st.markdown(
        f"""
        <div class="reference-card">
            <div class="reference-label">Complaint reference</div>
            <div class="reference-value">{reference}</div>
            <div class="reference-detail">Keep this reference ID to check your complaint status later.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_page_heading(
        "AI recommendation",
        "Initial routing and priority assessment",
        "These results support staff review. Campus staff make the final routing, priority, and resolution decisions.",
    )

    if duplicate_flag:
        st.warning("A potentially similar historic complaint was detected. Staff will review whether the reports are related.")
    else:
        st.success("No high-confidence duplicate was detected.")

    metric_col1, metric_col2, metric_col3 = st.columns(3, gap="medium")
    with metric_col1:
        st.metric(
            "Recommended category",
            result.get("predicted_category", "Not available"),
            f"{float(result.get('category_confidence', 0)):.1%} confidence",
        )
        st.caption(f"Department: {result.get('assigned_department', 'Not available')}")
    with metric_col2:
        st.metric(
            "Priority recommendation",
            f"{priority_emoji(priority)} {priority}",
            f"{float(result.get('priority_confidence', 0)):.1%} confidence",
        )
        st.caption("Recommendation only; staff verify final urgency.")
    with metric_col3:
        st.metric(
            "Estimated resolution",
            f"{float(result.get('estimated_resolution_hours', 0)):.1f} hrs",
            f"±{float(result.get('prediction_interval_plus_minus_hours', 0)):.1f} hrs",
        )
        st.caption("An estimate, not a service guarantee.")

    insight_col1, insight_col2 = st.columns(2, gap="large")
    with insight_col1:
        with st.container(border=True):
            st.markdown("#### Recommended department")
            st.write(result.get("assigned_department", "Not available"))
            st.caption("The category model uses the complaint text to suggest the most relevant operational department.")
    with insight_col2:
        with st.container(border=True):
            st.markdown("#### Duplicate review")
            duplicate_id = result.get("top_duplicate_id")
            duplicate_similarity = result.get("top_duplicate_similarity")
            if duplicate_flag:
                st.write("Staff review recommended")
                if duplicate_id:
                    detail = f"Similar historic reference: {duplicate_id}"
                    if duplicate_similarity is not None:
                        detail += f" · Similarity: {float(duplicate_similarity):.1%}"
                    st.caption(detail)
            else:
                st.write("No high-confidence match")
                st.caption("No similar historic complaint exceeded the review threshold.")

    with st.expander("Why did the system make this recommendation?"):
        st.write(result.get("explanation", "No prediction explanation is available."))
        st.markdown("##### Confidence and estimate guide")
        st.markdown(
            """
            - **Category confidence** indicates how strongly the classifier matched the complaint text to the suggested issue type.
            - **Priority confidence** indicates the model's confidence in the suggested urgency level.
            - **Resolution estimate** is a prediction, not a service guarantee; the ± value represents model uncertainty.
            - **Duplicate flag** indicates a semantically similar earlier complaint and requires staff review.
            """
        )

    action_col1, action_col2 = st.columns(2, gap="medium")
    with action_col1:
        st.button(
            "Track this complaint",
            type="primary",
            use_container_width=True,
            key="go_to_tracker",
            on_click=open_tracker,
            args=(reference,),
        )
    with action_col2:
        if st.button("Submit another complaint", use_container_width=True, key="start_new_complaint"):
            reset_complaint_form()
            st.rerun()


def make_light_chart(fig: Any) -> Any:
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#334155",
        title_font_color="#0F172A",
        margin=dict(l=10, r=10, t=55, b=15),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#0F172A"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0", zeroline=False, linecolor="#CBD5E1")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor="#CBD5E1")
    return fig


def show_staff_dashboard(api_ready: bool) -> None:
    if not is_staff_or_admin():
        st.error(
            "Please sign in with a Staff or Admin account "
            "to access the operations workspace."
        )
        return

    if not is_staff_or_admin():
        st.error(
            "Please sign in with a Staff or Admin account "
            "to access the operations workspace."
        )
        return

    render_page_heading(
        "Staff operations workspace",
        "Complaint queue and operational insights",
        "Review AI recommendations, prioritize urgent reports, and record staff decisions. Model outputs support—but do not replace—staff judgment.",
    )

    if not api_ready:
        st.error("FastAPI backend is unavailable. Start the API first.")
        return

    try:
        summary = api_get("/dashboard/summary")
    except requests.RequestException as error:
        st.error("Could not load the dashboard summary.")
        with st.expander("Technical details"):
            st.code(str(error))
        return

    metric_columns = st.columns(5, gap="medium")
    metric_columns[0].metric("Total complaints", summary.get("total_complaints", 0))
    metric_columns[1].metric("Open", summary.get("open_complaints", 0))
    metric_columns[2].metric("In progress", summary.get("in_progress_complaints", 0))
    metric_columns[3].metric("Critical active", summary.get("critical_open_complaints", 0))
    metric_columns[4].metric("Duplicate flags", summary.get("possible_duplicate_complaints", 0))

    status_col1, status_col2, status_col3 = st.columns(3, gap="medium")
    status_col1.info(f"Open queue: {summary.get('open_complaints', 0)} complaint(s) awaiting action.")
    status_col2.warning(f"Critical active: {summary.get('critical_open_complaints', 0)} complaint(s) need priority review.")
    status_col3.info(f"Possible duplicates: {summary.get('possible_duplicate_complaints', 0)} complaint(s) require verification.")

    st.markdown("### Filter the queue")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4, gap="medium")
    with filter_col1:
        status_filter = st.selectbox("Status", ["All"] + STATUSES, key="staff_status_filter")
    with filter_col2:
        priority_filter = st.selectbox("Priority", ["All"] + PRIORITIES, key="staff_priority_filter")
    with filter_col3:
        limit = st.selectbox("Records to display", [25, 50, 100, 200], index=1, key="staff_limit")
    with filter_col4:
        st.write("")
        refresh = st.button("Refresh queue", use_container_width=True, key="refresh_queue")

    if refresh:
        st.rerun()

    params: dict[str, Any] = {"limit": limit}
    if status_filter != "All":
        params["status"] = status_filter
    if priority_filter != "All":
        params["priority"] = priority_filter

    try:
        queue_data = api_get("/complaints", params=params)
    except requests.RequestException as error:
        st.error("Could not load the complaint queue.")
        with st.expander("Technical details"):
            st.code(str(error))
        return

    complaints = queue_data.get("complaints", [])
    df = pd.DataFrame(complaints)
    st.caption(f"Showing {len(df)} of {queue_data.get('total', len(df))} matched complaints.")

    if df.empty:
        st.info("No complaints match the selected filters.")
        return

    required_columns = {
        "complaint_reference",
        "predicted_priority",
        "status",
        "predicted_category",
        "assigned_department",
        "specific_location",
        "possible_duplicate",
        "estimated_resolution_hours",
        "created_at",
        "complaint_text",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        st.error("The complaint API response is missing required fields for the staff dashboard.")
        with st.expander("Missing fields"):
            st.code(", ".join(sorted(missing_columns)))
        return

    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        priority_counts = df["predicted_priority"].value_counts().reindex(PRIORITIES, fill_value=0).reset_index()
        priority_counts.columns = ["Priority", "Count"]
        priority_chart = px.bar(
            priority_counts,
            x="Priority",
            y="Count",
            color="Priority",
            color_discrete_map={
                "Low": "#22C55E",
                "Medium": "#EAB308",
                "High": "#F97316",
                "Critical": "#EF4444",
            },
            title="Priority distribution",
        )
        priority_chart.update_layout(showlegend=False)
        st.plotly_chart(make_light_chart(priority_chart), use_container_width=True)

    with chart_col2:
        department_counts = df["assigned_department"].value_counts().reset_index()
        department_counts.columns = ["Department", "Count"]
        department_chart = px.bar(
            department_counts,
            x="Count",
            y="Department",
            orientation="h",
            color="Count",
            color_continuous_scale=["#DBEAFE", "#2563EB"],
            title="Department workload",
        )
        department_chart.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(make_light_chart(department_chart), use_container_width=True)

    st.markdown("### Complaint queue")
    display_columns = [
        "complaint_reference",
        "predicted_priority",
        "status",
        "predicted_category",
        "assigned_department",
        "specific_location",
        "possible_duplicate",
        "estimated_resolution_hours",
        "created_at",
    ]
    display_df = df[display_columns].copy()
    display_df["estimated_resolution_hours"] = pd.to_numeric(
        display_df["estimated_resolution_hours"], errors="coerce"
    ).round(1)
    display_df["created_at"] = pd.to_datetime(
        display_df["created_at"], errors="coerce"
    ).dt.strftime("%d %b %Y, %H:%M")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "complaint_reference": "Reference",
            "predicted_priority": "Priority",
            "status": "Status",
            "predicted_category": "Category",
            "assigned_department": "Department",
            "specific_location": "Location",
            "possible_duplicate": "Duplicate flag",
            "estimated_resolution_hours": "Est. hours",
            "created_at": "Created",
        },
    )

    st.markdown("### Review and update a complaint")
    references = df["complaint_reference"].tolist()
    selected_reference = st.selectbox(
        "Select complaint reference",
        options=references,
        key="selected_complaint_reference",
    )
    selected = df.loc[df["complaint_reference"] == selected_reference].iloc[0].to_dict()

    with st.container(border=True):
        st.markdown(f"#### {selected_reference}")
        st.write(selected.get("complaint_text", "Complaint text is not available."))
        st.caption(
            f"Category: {selected.get('predicted_category', 'Not available')} · "
            f"Department: {selected.get('assigned_department', 'Not available')} · "
            f"Location: {selected.get('specific_location', 'Not specified')}"
        )

        detail_col1, detail_col2, detail_col3 = st.columns(3, gap="medium")
        selected_priority = selected.get("predicted_priority", "Not available")
        detail_col1.metric("Priority", f"{priority_emoji(selected_priority)} {selected_priority}")
        detail_col2.metric("Resolution estimate", f"{float(selected.get('estimated_resolution_hours', 0)):.1f} hrs")
        detail_col3.metric("Duplicate review", "Yes" if selected.get("possible_duplicate") else "No")

        duplicate_id = selected.get("top_duplicate_id")
        duplicate_similarity = selected.get("top_duplicate_similarity")
        if selected.get("possible_duplicate"):
            if duplicate_id:
                similarity_text = (
                    f" Similarity: {float(duplicate_similarity):.1%}."
                    if duplicate_similarity is not None
                    else ""
                )
                st.warning(
                    f"Duplicate review required: a similar historic complaint ({duplicate_id}) was identified.{similarity_text}"
                )
            else:
                st.warning("Duplicate review required: a potentially similar historic complaint was detected.")

        with st.form("staff_update_form"):
            current_status = selected.get("status", "Open")
            status_index = STATUSES.index(current_status) if current_status in STATUSES else 0
            new_status = st.selectbox("Update status", STATUSES, index=status_index)
            staff_notes = st.text_area(
                "Staff notes",
                value=selected.get("staff_notes") or "",
                placeholder="Example: Technician assigned. Expected inspection: tomorrow morning.",
                height=105,
            )
            review_confirmed = st.checkbox(
                "I reviewed the complaint details and confirm this update.",
                key=f"review_confirmed_{selected_reference}",
            )
            update_submitted = st.form_submit_button(
                "Save staff update",
                type="primary",
                use_container_width=True,
            )

        if update_submitted:
            if not review_confirmed:
                st.warning("Confirm that you reviewed the complaint before saving the staff update.")
                return

            update_payload = {
                "status": new_status,
                "staff_notes": staff_notes.strip() or None,
            }
            try:
                with st.spinner("Saving staff update..."):
                    api_patch(f"/complaints/{selected_reference}", update_payload)
                st.success(f"{selected_reference} was updated to '{new_status}'.")
                st.rerun()
            except requests.RequestException as error:
                st.error("The staff update could not be saved. Please try again.")
                with st.expander("Technical details"):
                    st.code(str(error))


def show_complaint_tracker(api_ready: bool) -> None:
    if st.session_state.get("pending_track_reference"):
        st.session_state["track_reference"] = st.session_state.pop(
            "pending_track_reference"
        )

    if not is_authenticated():
        st.info(
            "Please sign in to view complaint status and timeline information."
        )
        return

    render_page_heading(
        "Complaint tracking",
        "Check your complaint status",
        "Enter the reference ID received after submission to view progress, routing details, and the latest staff update.",
    )

    if not api_ready:
        st.error("The complaint service is unavailable. Please start the FastAPI backend and try again.")
        return

    with st.container(border=True):
        with st.form("complaint_tracker_form"):
            tracker_col1, tracker_col2 = st.columns([4, 1], gap="medium")
            with tracker_col1:
                complaint_reference = st.text_input(
                    "Complaint reference",
                    key="track_reference",
                    placeholder="Example: CR-20260830-0001",
                    max_chars=100,
                    help="Use the reference shown after you submitted your complaint.",
                )
            with tracker_col2:
                st.write("")
                lookup_submitted = st.form_submit_button(
                    "Track complaint",
                    type="primary",
                    use_container_width=True,
                )

    if lookup_submitted:
        reference = complaint_reference.strip().upper()
        if not reference:
            st.error("Enter a complaint reference to continue.")
            return
        try:
            with st.spinner("Fetching complaint status..."):
                st.session_state["tracked_complaint"] = api_get(f"/complaints/{reference}")
        except requests.HTTPError as error:
            st.session_state["tracked_complaint"] = None
            if error.response is not None and error.response.status_code == 404:
                st.warning("No complaint was found for that reference. Check the ID and try again.")
            else:
                st.error("We could not retrieve this complaint. Please try again.")
                with st.expander("Technical details"):
                    st.code(str(error))
        except requests.RequestException as error:
            st.session_state["tracked_complaint"] = None
            st.error("The complaint service could not be reached. Please try again shortly.")
            with st.expander("Technical details"):
                st.code(str(error))

    complaint = st.session_state.get("tracked_complaint")
    if not complaint:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Track with your reference ID</div>
                <div class="info-card-body">
                    Your reference is created when you submit a complaint. It lets you view the latest status,
                    department assignment, recommendation summary, and staff note.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    reference = complaint.get("complaint_reference", "Not available")
    status = complaint.get("status", "Open")
    priority = complaint.get("predicted_priority", "Not available")
    is_duplicate = bool(complaint.get("possible_duplicate"))

    st.markdown(
        f"""
        <div class="reference-card">
            <div class="reference-label">Complaint reference</div>
            <div class="reference-value">{reference}</div>
            <div class="reference-detail">Current status: {status_badge(status)} &nbsp; Last updated: {format_datetime(complaint.get('updated_at'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4, gap="medium")
    metric_col1.metric("Current status", status)
    metric_col2.metric(
        "Priority recommendation",
        f"{priority_emoji(priority)} {priority}",
        f"{float(complaint.get('priority_confidence', 0)):.1%} confidence",
    )
    metric_col3.metric(
        "Resolution estimate",
        f"{float(complaint.get('estimated_resolution_hours', 0)):.1f} hrs",
        f"±{float(complaint.get('prediction_interval_plus_minus_hours', 0)):.1f} hrs",
    )
    metric_col4.metric(
        "Duplicate review",
        "Review needed" if is_duplicate else "No match flagged",
        "Potentially similar report" if is_duplicate else "No high-confidence match",
    )

    details_col1, details_col2 = st.columns(2, gap="large")
    with details_col1:
        with st.container(border=True):
            st.markdown("#### Routing recommendation")
            st.write(f"**Category:** {complaint.get('predicted_category', 'Not available')}")
            st.write(f"**Assigned department:** {complaint.get('assigned_department', 'Not available')}")
            st.write(f"**Location:** {complaint.get('specific_location', 'Not specified')}")
            st.write(f"**Language:** {complaint.get('language', 'Not available')}")
    with details_col2:
        with st.container(border=True):
            st.markdown("#### Complaint details")
            st.write(f"**People affected:** {complaint.get('affected_population', 'Not available')}")
            safety_flag = complaint.get("safety_flag")
            safety_label = "Yes" if safety_flag else "No" if safety_flag is not None else "Not available"
            st.write(f"**Safety concern:** {safety_label}")
            st.write(f"**Earlier reports:** {complaint.get('repeat_count', 'Not available')}")
            st.write(f"**Submitted:** {format_datetime(complaint.get('created_at'))}")

    st.markdown("#### Your complaint")
    st.info(complaint.get("complaint_text", "Complaint text is not available."))

    if complaint.get("staff_notes"):
        st.markdown("#### Latest staff update")
        st.success(complaint["staff_notes"])
    else:
        st.caption("No staff update has been added yet.")

    with st.expander("Why did the system make this recommendation?"):
        st.write(complaint.get("explanation", "No prediction explanation is available for this complaint."))

    if is_duplicate and complaint.get("top_duplicate_id"):
        similarity = complaint.get("top_duplicate_similarity")
        suffix = f" Similarity: {float(similarity):.1%}." if similarity is not None else ""
        st.warning(
            f"A similar historic complaint was identified ({complaint['top_duplicate_id']}). "
            f"Staff will review whether the reports are related.{suffix}"
        )

    st.caption(
        "CampusResolve-AI provides decision-support recommendations. Campus staff make the final routing, priority, and resolution decisions."
    )


def show_about_page() -> None:
    render_page_heading(
        "About the system",
        "Transparent AI for campus support",
        "CampusResolve-AI is an academic ML/NLP research prototype for more consistent complaint routing and tracking.",
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.markdown("#### What the platform does")
            st.markdown(
                """
                - Accepts complaints in English, Tamil, and Tamil-English (Tanglish)
                - Recommends an issue category and responsible department
                - Detects potentially similar historic complaints
                - Forecasts a priority level and likely resolution time
                - Gives students a reference ID for status tracking
                """
            )
    with col2:
        with st.container(border=True):
            st.markdown("#### Important guidance")
            st.markdown(
                """
                - AI outputs support staff review; they are not final decisions
                - Do not enter passwords, government IDs, phone numbers, or sensitive personal information
                - For fire, injury, severe electrical hazards, or urgent security threats, contact campus emergency support directly
                """
            )

    st.info("This research prototype uses synthetic complaint data for academic demonstration and evaluation.")


def initialise_session_state() -> None:
    defaults = {
        "last_submission": None,
        "complaint_text": "",
        "language": "Tamil-English / Tanglish",
        "location_type": "Hostel",
        "specific_location": "Hostel Block B",
        "affected_population": 1,
        "safety_flag": False,
        "repeat_count": 0,
        "active_page": "Report an issue",
        "track_reference": "",
        "tracked_complaint": None,
        "pending_track_reference": None,
        "requested_page": None,
        "access_token": None,
        "current_user": None,
        "auth_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(
        page_title="CampusResolve-AI",
        page_icon="🏫",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    initialise_session_state()
    api_ready, api_status = get_api_health()

    if st.session_state.get("requested_page"):
        st.session_state["active_page"] = st.session_state.pop("requested_page")

    st.markdown(
        """
        <div class="topbar">
            <div>
                <div class="eyebrow">Student support portal</div>
                <div class="topbar-title">CampusResolve-AI</div>
            </div>
            <div class="topbar-status">
                <span class="status-dot"></span>
                AI-assisted complaint management
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        
        st.markdown(
            """
            <div class="brand-block">
                <div class="brand-name">CampusResolve-AI</div>
                <div class="brand-caption">Student support and staff operations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if api_ready:
            st.success("Backend connected")
        else:
            st.error("Backend unavailable")
        st.caption(f"API status: {api_status}")

        st.divider()
        render_auth_panel()

        st.divider()
        st.markdown("### Navigation")

        available_pages = {
            "Report an issue": "report",
            "Track complaint": "track",
            "About": "about",
        }

        if is_staff_or_admin():
            available_pages["Staff workspace"] = "staff"

        if st.session_state["active_page"] not in available_pages:
            st.session_state["active_page"] = "Report an issue"

        page = st.radio(
            "Navigation",
            options=list(available_pages.keys()),
            key="active_page",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### Demo samples")
        example_name = st.selectbox(
            "Load a sample complaint",
            options=["Choose a sample"] + list(EXAMPLES.keys()),
            key="example_name",
        )
        if st.button("Load sample", use_container_width=True, key="load_sample"):
            if example_name == "Choose a sample":
                st.info("Select a sample complaint first.")
            else:
                apply_example(EXAMPLES[example_name])
                st.session_state["requested_page"] = "Report an issue"
                st.rerun()

        st.divider()
        st.markdown("### Safety notice")
        st.caption(
            "For immediate fire, injury, severe electrical risk, or urgent security threats, contact campus emergency support directly."
        )
        with st.expander("Developer details"):
            st.caption(f"Backend: {API_BASE_URL}")
            st.caption(f"API status: {api_status}")
        st.divider()
        st.caption("Academic research prototype · Staff review is required.")

    page_key = available_pages[page]

    if page_key == "report":
        show_student_portal(api_ready)
    elif page_key == "track":
        show_complaint_tracker(api_ready)
    elif page_key == "staff":
        show_staff_dashboard(api_ready)
    else:
        show_about_page()

    st.markdown(
        """
        <div class="footer">
            CampusResolve-AI · Explainable AI-assisted campus complaint management
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
