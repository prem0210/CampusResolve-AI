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

EXAMPLES = {
    "💧 Water leakage": {
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
    "📶 Wi-Fi unavailable": {
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
    "🪑 Classroom facility": {
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
    st.html(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 10% -5%, rgba(56, 189, 248, 0.19), transparent 32%),
                    radial-gradient(circle at 100% 0%, rgba(167, 139, 250, 0.15), transparent 30%),
                    #07111f;
            }

            [data-testid="stHeader"] {
                background: rgba(7, 17, 31, 0.45);
                backdrop-filter: blur(18px);
            }

            .block-container {
                max-width: 1240px;
                padding-top: 1.3rem;
                padding-bottom: 3.5rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #09172a, #050e1b);
                border-right: 1px solid rgba(125, 211, 252, 0.14);
            }

            .hero {
                padding: 2.2rem 2.4rem;
                border: 1px solid rgba(125, 211, 252, 0.24);
                border-radius: 26px;
                background: linear-gradient(115deg, #0a1d33, #12315a);
                box-shadow: 0 22px 65px rgba(0, 0, 0, 0.30);
                margin-bottom: 1.3rem;
            }

            .badge {
                display: inline-block;
                font-size: 0.75rem;
                font-weight: 800;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                color: #bcecff;
                background: rgba(56, 189, 248, 0.13);
                border: 1px solid rgba(56, 189, 248, 0.30);
                padding: 0.38rem 0.75rem;
                border-radius: 999px;
                margin-bottom: 0.85rem;
            }

            .hero-title {
                color: #ffffff;
                font-size: 2.45rem;
                font-weight: 850;
                letter-spacing: -0.045em;
                line-height: 1.05;
            }

            .hero-highlight {
                background: linear-gradient(90deg, #67e8f9, #c4b5fd);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-subtitle {
                color: #c1d8e9;
                font-size: 1.02rem;
                margin-top: 0.9rem;
                line-height: 1.6;
            }

            .section-kicker {
                color: #67e8f9;
                font-size: 0.76rem;
                font-weight: 850;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin-bottom: 0.25rem;
            }

            .section-title {
                color: #f0f9ff;
                font-size: 1.45rem;
                font-weight: 800;
                letter-spacing: -0.025em;
                margin-bottom: 0.35rem;
            }

            .section-subtitle {
                color: #8fa8ba;
                margin-bottom: 1.15rem;
                font-size: 0.91rem;
            }

            [data-testid="stForm"] {
                background: rgba(14, 27, 46, 0.74);
                border: 1px solid rgba(125, 211, 252, 0.16);
                border-radius: 21px;
                padding: 1.25rem 1.25rem 0.65rem 1.25rem;
            }

            [data-testid="stMetric"] {
                background: linear-gradient(145deg, rgba(21, 49, 81, 0.80), rgba(10, 25, 44, 0.84));
                border: 1px solid rgba(125, 211, 252, 0.16);
                border-radius: 18px;
                padding: 1.05rem;
            }

            .queue-card {
                border: 1px solid rgba(125, 211, 252, 0.16);
                border-radius: 16px;
                padding: 1rem;
                margin-bottom: 0.9rem;
                background: rgba(7, 20, 37, 0.66);
            }

            .queue-reference {
                color: #7dd3fc;
                font-weight: 800;
                font-size: 1rem;
            }

            .queue-meta {
                color: #88a2b7;
                font-size: 0.82rem;
                margin-top: 0.35rem;
            }

            .footer {
                border-top: 1px solid rgba(125, 211, 252, 0.12);
                margin-top: 2.3rem;
                padding-top: 1.2rem;
                text-align: center;
                color: #6f8a9f;
                font-size: 0.8rem;
            }
        </style>
        """
    )


def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}{path}",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def api_patch(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.patch(
        f"{API_BASE_URL}{path}",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_api_health() -> tuple[bool, str]:
    try:
        data = api_get("/health")
        return bool(data.get("models_loaded")), str(data.get("status"))
    except requests.RequestException:
        return False, "unreachable"


def priority_emoji(priority: str) -> str:
    return {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🟠",
        "Critical": "🔴",
    }.get(priority, "⚪")


def apply_example(example: dict[str, Any]) -> None:
    for key, value in example.items():
        st.session_state[key] = value


def show_student_portal(api_ready: bool) -> None:
    st.markdown(
        '<div class="section-kicker">Student complaint portal</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Tell us what happened</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        "Submit in English, Tamil, or Tamil-English (Tanglish). "
        "The system will provide an explainable recommendation."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("complaint_form", clear_on_submit=False):
        complaint_text = st.text_area(
            "Complaint details",
            placeholder=(
                "Example: Hostel Block B la water leak aagudhu and floor "
                "slippery ah iruku."
            ),
            height=150,
            key="complaint_text",
        )

        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            language = st.selectbox(
                "Language",
                options=list(LANGUAGES.keys()),
                key="language",
            )
            location_type = st.selectbox(
                "Location type",
                options=LOCATION_TYPES,
                key="location_type",
            )

        with col2:
            specific_location = st.text_input(
                "Specific location",
                max_chars=150,
                key="specific_location",
            )
            affected_population = st.number_input(
                "People affected",
                min_value=1,
                max_value=10000,
                step=1,
                key="affected_population",
            )

        with col3:
            safety_flag = st.checkbox(
                "Safety / health concern",
                key="safety_flag",
            )
            repeat_count = st.number_input(
                "Earlier reports",
                min_value=0,
                max_value=100,
                step=1,
                key="repeat_count",
            )

        submitted = st.form_submit_button(
            "✨ Submit and analyze complaint",
            type="primary",
            use_container_width=True,
            disabled=not api_ready,
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
                result = api_post("/complaints", payload)

            st.session_state.last_submission = result
            st.success(
                f"Complaint submitted successfully. Reference: "
                f"{result['complaint_reference']}"
            )
        except requests.RequestException as error:
            st.error(f"Submission failed: {error}")

    result = st.session_state.get("last_submission")

    if not result:
        return

    st.divider()
    st.markdown("### AI recommendation")

    if result["possible_duplicate"]:
        st.warning("A possible duplicate was detected. Staff will review it.")
    else:
        st.success("No high-confidence duplicate was detected.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Category",
            result["predicted_category"],
            f"{result['category_confidence']:.1%} confidence",
        )
        st.caption(f"Department: {result['assigned_department']}")

    with col2:
        st.metric(
            "Priority",
            f"{priority_emoji(result['predicted_priority'])} "
            f"{result['predicted_priority']}",
            f"{result['priority_confidence']:.1%} confidence",
        )

    with col3:
        st.metric(
            "Resolution estimate",
            f"{result['estimated_resolution_hours']:.1f} hrs",
            f"±{result['prediction_interval_plus_minus_hours']:.1f} hrs",
        )

    st.info(result["explanation"])


def show_staff_dashboard(api_ready: bool) -> None:
    st.markdown(
        '<div class="section-kicker">Staff operations workspace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Complaint queue and operational insights</div>',
        unsafe_allow_html=True,
    )

    if not api_ready:
        st.error("FastAPI backend is unavailable. Start the API first.")
        return

    try:
        summary = api_get("/dashboard/summary")
    except requests.RequestException as error:
        st.error(f"Could not load dashboard summary: {error}")
        return

    metric_columns = st.columns(5)
    metric_columns[0].metric("Total", summary["total_complaints"])
    metric_columns[1].metric("Open", summary["open_complaints"])
    metric_columns[2].metric("In progress", summary["in_progress_complaints"])
    metric_columns[3].metric("Critical active", summary["critical_open_complaints"])
    metric_columns[4].metric("Duplicate flags", summary["possible_duplicate_complaints"])

    st.write("")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        status_filter = st.selectbox(
            "Status",
            options=["All"] + STATUSES,
            key="staff_status_filter",
        )

    with filter_col2:
        priority_filter = st.selectbox(
            "Priority",
            options=["All"] + PRIORITIES,
            key="staff_priority_filter",
        )

    with filter_col3:
        limit = st.selectbox(
            "Records to display",
            options=[25, 50, 100, 200],
            index=1,
            key="staff_limit",
        )

    with filter_col4:
        refresh = st.button("🔄 Refresh queue", use_container_width=True)

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
        st.error(f"Could not load complaint queue: {error}")
        return

    complaints = queue_data["complaints"]
    df = pd.DataFrame(complaints)

    st.caption(f"Showing {len(df)} of {queue_data['total']} matched complaints.")

    if df.empty:
        st.info("No complaints match the selected filters.")
        return

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        priority_counts = (
            df["predicted_priority"]
            .value_counts()
            .reindex(PRIORITIES, fill_value=0)
            .reset_index()
        )
        priority_counts.columns = ["Priority", "Count"]

        priority_chart = px.bar(
            priority_counts,
            x="Priority",
            y="Count",
            color="Priority",
            color_discrete_map={
                "Low": "#22c55e",
                "Medium": "#eab308",
                "High": "#f97316",
                "Critical": "#ef4444",
            },
            title="Priority distribution",
        )
        priority_chart.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d7edf9",
        )
        st.plotly_chart(priority_chart, use_container_width=True)

    with chart_col2:
        department_counts = (
            df["assigned_department"]
            .value_counts()
            .reset_index()
        )
        department_counts.columns = ["Department", "Count"]

        department_chart = px.bar(
            department_counts,
            x="Count",
            y="Department",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            title="Department workload",
        )
        department_chart.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d7edf9",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(department_chart, use_container_width=True)

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
    display_df["estimated_resolution_hours"] = (
        display_df["estimated_resolution_hours"].round(1)
    )
    display_df["created_at"] = pd.to_datetime(
        display_df["created_at"]
    ).dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "complaint_reference": "Reference",
            "predicted_priority": "Priority",
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

    selected = df[
        df["complaint_reference"] == selected_reference
    ].iloc[0].to_dict()

    with st.container(border=True):
        st.markdown(
            f'<div class="queue-reference">{selected["complaint_reference"]}</div>',
            unsafe_allow_html=True,
        )
        st.write(selected["complaint_text"])

        st.caption(
            f"Category: {selected['predicted_category']} · "
            f"Department: {selected['assigned_department']} · "
            f"Location: {selected['specific_location']}"
        )

        detail_col1, detail_col2, detail_col3 = st.columns(3)
        detail_col1.metric(
            "Priority",
            f"{priority_emoji(selected['predicted_priority'])} "
            f"{selected['predicted_priority']}",
        )
        detail_col2.metric(
            "Resolution estimate",
            f"{selected['estimated_resolution_hours']:.1f} hrs",
        )
        detail_col3.metric(
            "Duplicate risk",
            "Yes" if selected["possible_duplicate"] else "No",
        )

        if selected["top_duplicate_id"]:
            st.warning(
                f"Top similar historic complaint: "
                f"{selected['top_duplicate_id']} "
                f"(similarity: {selected['top_duplicate_similarity']:.3f})"
            )

        with st.form("staff_update_form"):
            new_status = st.selectbox(
                "Update status",
                options=STATUSES,
                index=STATUSES.index(selected["status"]),
            )

            staff_notes = st.text_area(
                "Staff notes",
                value=selected["staff_notes"] or "",
                placeholder="Example: Maintenance technician assigned.",
                height=100,
            )

            update_submitted = st.form_submit_button(
                "Save staff update",
                type="primary",
            )

        if update_submitted:
            update_payload = {
                "status": new_status,
                "staff_notes": staff_notes.strip() or None,
            }

            try:
                api_patch(
                    f"/complaints/{selected_reference}",
                    update_payload,
                )
                st.success("Complaint status and staff notes updated.")
                st.rerun()
            except requests.RequestException as error:
                st.error(f"Update failed: {error}")


def main() -> None:
    st.set_page_config(
        page_title="CampusResolve-AI",
        page_icon="🏫",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    defaults = {
        "last_submission": None,
        "complaint_text": "",
        "language": "Tamil-English / Tanglish",
        "location_type": "Hostel",
        "specific_location": "Hostel Block B",
        "affected_population": 1,
        "safety_flag": False,
        "repeat_count": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    api_ready, api_status = get_api_health()

    st.markdown(
        """
        <div class="hero">
            <div class="badge">Explainable multilingual campus AI</div>
            <div class="hero-title">
                Campus complaints,<br>
                <span class="hero-highlight">resolved with intelligence.</span>
            </div>
            <div class="hero-subtitle">
                A multilingual AI platform for complaint routing, duplicate
                awareness, priority forecasting, resolution-time estimation,
                and staff operations management.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## 🏫 CampusResolve-AI")

        if api_ready:
            st.success("● Backend connected")
        else:
            st.error("● Backend unavailable")

        st.caption(f"API status: {api_status}")
        st.caption(f"Backend: {API_BASE_URL}")

        st.divider()
        st.markdown("### Demo samples")

        example_name = st.selectbox(
            "Load a student example",
            options=["Choose a sample"] + list(EXAMPLES.keys()),
        )

        if st.button("Load sample", use_container_width=True):
            if example_name != "Choose a sample":
                apply_example(EXAMPLES[example_name])
                st.rerun()

        st.divider()
        st.markdown("### Safety notice")
        st.caption(
            "For immediate fire, injury, severe electrical risk, or urgent "
            "security threats, contact campus emergency support directly."
        )

        st.divider()
        st.caption("Research prototype · Staff review is required.")

    student_tab, staff_tab = st.tabs(
        ["📝 Student Complaint Portal", "📊 Staff Operations Dashboard"]
    )

    with student_tab:
        show_student_portal(api_ready)

    with staff_tab:
        show_staff_dashboard(api_ready)

    st.markdown(
        """
        <div class="footer">
            CampusResolve-AI · Individual ML/NLP Research Project ·
            Explainable decision support for campus complaint handling
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()