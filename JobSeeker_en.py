import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import AzureOpenAI

# ========== Azure OpenAI Initialization ==========
client = AzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",
    api_version="2024-06-01",
    api_key="da67de83ae6f4b82985c7978a1e83c64"
)

def get_response(message):
    """
    A function to get chatbot responses from Azure OpenAI.
    It only replies to job-related keywords, ignoring unrelated queries.
    """
    job_keywords = [
        # Common English job-related keywords
        "job", "jobs", "role", "position", "vacancy", "offer", "offer negotiation",
        "intern", "internship", "salary", "compensation", "pay", "benefits",
        "resume", "cv", "cover letter", "headhunter", "application", "apply",
        "hiring", "hire", "recruit", "recruiter", "recruiting", "recruitment",
        "employment", "unemployment", "termination", "contract", "employment contract",
        "agreement", "onboarding", "offboarding", "fired", "layoff", "furlough",
        "redundancy", "sabbatical", "work visa", "work permit", "career",
        "career path", "professional development", "job description", "jd", "job duty",
        "freelance", "freelancer", "gig", "gig economy", "promotion", "demotion",
        "raise", "bonus", "stock option", "esop", "human resources", "hr",
        # Common recruiting platforms
        "linkedin", "indeed", "glassdoor", "monster",
    ]
    lower_msg = message.lower()
    if not any(kw in lower_msg for kw in job_keywords):
        return "Sorry, I only respond to job or career-related questions."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1,
        messages=[
            {"role": "system", "content": "You are an AI assistant who only answers job and career-related questions. Please refuse to answer other topics."},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content


# ========== Streamlit Main App ==========

@st.cache_data
def load_data():
    """
    Loads the English-version Excel data for job listings.
    Make sure the file path and column names match your dataset structure.
    """
    file_path = '/Users/conghaoji/Desktop/Recruitment_Data_English.xlsx'
    return pd.read_excel(file_path)

@st.cache_resource
def load_embedding_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Loads the sentence transformer model for encoding text.
    (Switched to an English-compatible model for demonstration.)
    """
    return SentenceTransformer(model_name)

@st.cache_data
def compute_embeddings(df, _model):
    """
    Combines certain columns (Working city + Available positions + Company description)
    into a single string and encodes them with the Transformer model for similarity search.
    """
    df['combined_info'] = (
        df['Working city'].astype(str) + ' ' +
        df['Available positions'].astype(str) + ' ' +
        df['Company description'].astype(str)
    )
    embeddings = _model.encode(df['combined_info'].tolist(), show_progress_bar=True)
    return embeddings

import re

def display_job_info(df, page_num, items_per_page=10):
    """
    Displays job information in a paginated format.
    """
    total_items = len(df)
    if total_items == 0:
        st.info("No data available.")
        return

    total_pages = (total_items - 1) // items_per_page + 1
    if page_num < 1:
        page_num = 1
    elif page_num > total_pages:
        page_num = total_pages
        st.warning(f"Out of range. Jumping to last page: Page {total_pages}")

    start_idx = (page_num - 1) * items_per_page
    end_idx = start_idx + items_per_page
    subset = df.iloc[start_idx:end_idx]

    if subset.empty:
        st.warning("No results on this page, try another page number.")
        return

    # Pull the user_position_input from session_state
    user_position_input = st.session_state.get("user_position_input", "").strip().lower()

    # --- HTML/CSS for job cards ---
    st.markdown(""" 
    <head>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    </head>
    <style>
    .job-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 3px;
        margin-bottom: 3px;
        background-color: #fefefe;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: flex-start; 
    }
    .job-left {
        width: 75%;
    }
    .job-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 5px;
        color: #333;
    }
    .job-meta {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 8px;
    }
    .job-meta span { margin-right: 20px; }
    .company-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        border-radius: 4px;
        padding: 1px 6px;
        margin-right: 5px;
        font-size: 0.85rem;
    }
    .deadline-text {
        color: #d9534f;
    }
    .job-positions {
        margin-top: 5px;
        color: #333;
    }
    .pos-badge {
        display: inline-block;
        border-radius: 6px;
        padding: 4px 8px;
        margin: 4px 4px 0 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .pos-green {
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .pos-gray {
        border: 2px solid #a8a5a5;
        color: #a8a5a5;
    }
    .job-right {
        width: 23%;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 12px;
    }
    a.styled-btn-detail {
        display: inline-block;
        text-decoration: none;
        background-color: #ffffff;
        color: #333;
        font-size: 1.1rem;
        padding: 10px 18px;
        border-radius: 8px;
        border: 2px solid #a8a5a5;
        transition: background-color 0.2s ease;
    }
    a.styled-btn-apply {
        display: inline-block;
        text-decoration: none;
        background-color: #1976d2;
        color: #ffffff;
        font-size: 1.1rem;
        padding: 10px 18px;
        border-radius: 8px;
        border: 2px solid #1976d2;
        transition: background-color 0.2s ease;
    }
    a.styled-btn-detail:hover {
        background-color: #f0f0f0;
    }
    a.styled-btn-detail:active {
        background-color: #e0e0e0;
    }
    a.styled-btn-apply:hover {
        background-color: #1565c0;
    }
    a.styled-btn-apply:active {
        background-color: #0d47a1;
    }
    </style>
    """, unsafe_allow_html=True)

    current_date = datetime.now().date()

    for idx, row in subset.iterrows():
        detail_link = row.get('Detail link', '#')
        apply_link = row.get('Application link', '#')

        tags = row['Company label']
        if isinstance(tags, list):
            tags_html = "".join(f'<span class="company-tag">{t}</span>' for t in tags)
        else:
            tags_html = f"<span class='company-tag'>{tags}</span>" if tags else ""

        # Deadline check
        expired = False
        if pd.notnull(row['Deadline']):
            if row['Deadline'].date() < current_date:
                expired = True
            deadline_str = row['Deadline'].date().strftime("%Y-%m-%d")
        else:
            deadline_str = "No Info"

        if expired:
            date_html = f'<span style="color:red;"> <i class="fas fa-clock"></i> {deadline_str} (Expired)</span>'
        else:
            date_html = f'<span style="color:black;"> <i class="fas fa-clock"></i> {deadline_str}</span>'

        with st.container():
            st.markdown('<div class="job-card">', unsafe_allow_html=True)
            st.markdown('<div class="job-left">', unsafe_allow_html=True)

            # 1) Company Name
            st.markdown(f'<div class="job-title">{row["Company Name"]}</div>', unsafe_allow_html=True)
            # 2) City / Tags / Deadline
            st.markdown(
                f'<div class="job-meta">'
                f'<span><i class="fas fa-map-marker-alt"></i> {row["Working city"]}</span>'
                f'<span><i class="fas fa-tags"></i> {tags_html}</span>'
                f'{date_html}'
                f'</div>',
                unsafe_allow_html=True
            )

            # 3) Available positions
            positions_raw = str(row["Available positions"]) if row["Available positions"] else ""
            # We originally split by "-" to get each position:
            positions = re.split(r'[-]+', positions_raw.strip())

            pos_html_list = []
            for p in positions:
                p_clean = p.strip()
                if not p_clean:
                    continue
                # Make it case-insensitive:
                if user_position_input and (user_position_input in p_clean.lower()):
                    pos_html_list.append(f'<span class="pos-badge pos-green">{p_clean}</span>')
                else:
                    pos_html_list.append(f'<span class="pos-badge pos-gray">{p_clean}</span>')

            pos_html = " ".join(pos_html_list)
            st.markdown(f'<div class="job-positions">{pos_html}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # close job-left

            # Right-side buttons
            st.markdown('<div class="job-right">', unsafe_allow_html=True)
            st.markdown(f"""
                <a class="styled-btn-detail" href="{detail_link}" target="_blank">Detail Link</a>
                <a class="styled-btn-apply" href="{apply_link}" target="_blank">Application Link</a>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) 

            st.markdown('</div>', unsafe_allow_html=True)

    return page_num, total_pages


def main():
    st.set_page_config(page_title="Job Seeker Assistant (Beta)", layout="wide")

    # App layout
    col1, col2 = st.columns([3,2], gap="large")

    with col1:
        st.title("🌟Job Seeker Smart Delivery Assistant (Beta)")

        with st.spinner('Loading recruitment data, please wait...'):
            data = load_data()
            time.sleep(1)

        global model
        model = load_embedding_model()

        if 'exact_matches' not in st.session_state:
            st.session_state['exact_matches'] = pd.DataFrame()
        if 'similar_matches' not in st.session_state:
            st.session_state['similar_matches'] = pd.DataFrame()

        # --- Parse Company Labels (Fix: only split by comma) ---
        # If a row has "Media/live broadcast, state-owned enterprises", it means two tags:
        #   ["Media/live broadcast", "state-owned enterprises"].
        all_tags = set()
        parsed_tags = []
        for idx, row in data.iterrows():
            raw = str(row.get('Company label', '')).strip()
            if raw and raw.lower() != 'nan':
                # Split ONLY by comma, so "Media/live broadcast" remains a single tag
                tag_list = [t.strip() for t in raw.split(',') if t.strip()]
            else:
                tag_list = []
            parsed_tags.append(tag_list)
            all_tags.update(tag_list)
        data['Company label'] = parsed_tags
        all_tags = sorted(list(all_tags))

        with st.expander("🔍 Search Filters", expanded=True):
            city_input = st.text_input("City:", "Nationwide")
            position_input = st.text_input("Position:", "data analysis")
            st.session_state["user_position_input"] = position_input  # store user input (lowercase match used later)

            selected_tags = st.multiselect("Filter by Company Label (Multiple Select)", options=all_tags)

            if st.button("Search Jobs"):
                if selected_tags:
                    # If any selected tag appears in the company's label list, we keep it
                    mask = data['Company label'].apply(lambda tags: any(t in tags for t in selected_tags))
                    filtered_data = data[mask].copy()
                else:
                    filtered_data = data.copy()

                data_embeddings = compute_embeddings(filtered_data, model)
                user_query = f"{city_input} {position_input}"
                query_embedding = model.encode([user_query.lower()], show_progress_bar=False)
                similarity_scores = cosine_similarity(query_embedding, data_embeddings).flatten()
                filtered_data['similarity'] = similarity_scores

                # Make city & position match case-insensitive
                city_lower = city_input.lower().strip()
                position_lower = position_input.lower().strip()

                filtered_data['city_match'] = filtered_data['Working city'].astype(str).apply(
                    lambda x: city_lower in x.lower()
                )
                filtered_data['position_match'] = filtered_data['Available positions'].astype(str).apply(
                    lambda x: position_lower in x.lower()
                )
                filtered_data['city_position_match'] = filtered_data['city_match'] & filtered_data['position_match']

                # Adjust similarity
                def compute_adjusted_similarity(row):
                    base_sim = row['similarity']
                    if row['city_position_match']:
                        return base_sim * 1.2
                    elif row['city_match'] or row['position_match']:
                        return base_sim * 1.1
                    else:
                        return base_sim

                filtered_data['adjusted_similarity'] = filtered_data.apply(compute_adjusted_similarity, axis=1)

                # Deadline effect
                filtered_data['Deadline'] = pd.to_datetime(filtered_data['Deadline'], errors='coerce')
                current_date = datetime.now().date()
                filtered_data['valid_date'] = filtered_data['Deadline'].apply(
                    lambda x: x and x.date() >= current_date if pd.notnull(x) else True
                )
                # If deadline is expired, reduce similarity
                filtered_data['adjusted_similarity'] = filtered_data.apply(
                    lambda row: row['adjusted_similarity'] * 0.5 if not row['valid_date'] else row['adjusted_similarity'],
                    axis=1
                )

                filtered_data = filtered_data.sort_values(by='adjusted_similarity', ascending=False)

                exact_matches = filtered_data[filtered_data['city_position_match']].copy()
                similar_matches = filtered_data[~filtered_data['city_position_match']].copy()

                st.session_state['exact_matches'] = exact_matches
                st.session_state['similar_matches'] = similar_matches

        exact_matches = st.session_state['exact_matches']
        similar_matches = st.session_state['similar_matches']

        if exact_matches.empty and similar_matches.empty:
            st.info("Please set your filters and click 'Search Jobs'.")
        else:
            tab1, tab2 = st.tabs(["Exact Match", "Similar Match"])
            with tab1:
                if exact_matches.empty:
                    st.warning("No exact matches found.")
                else:
                    st.subheader("Exact Matches:")
                    exact_total = len(exact_matches)
                    exact_pages = (exact_total - 1) // 10 + 1
                    exact_page_num = st.number_input(
                        f"Exact matches, total {exact_total}. Choose page:",
                        min_value=1,
                        max_value=exact_pages,
                        value=1,
                        step=1,
                        key="exact_matches_page"
                    )
                    display_job_info(exact_matches, exact_page_num, items_per_page=10)

            with tab2:
                st.subheader("You may also be interested in (Top 20):")
                similar_top20 = similar_matches.head(20)
                if similar_top20.empty:
                    st.write("No similar job recommendations.")
                else:
                    st.write(f"Total {len(similar_matches)} similar matches. Showing top 20 below:")
                    display_job_info(similar_top20, 1, items_per_page=20)

    # ========== Right Column: AI Chatbox ==========
    with col2:
        st.header("🤖 AI Career Consulting Chat")

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "chat_input" not in st.session_state:
            st.session_state["chat_input"] = ""

        # Display chat
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat_item in st.session_state["chat_history"]:
            role = chat_item["role"]
            content = chat_item["content"]
            if role == "user":
                st.markdown(f'<div class="chat-message-user">👤 User: {content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-ai">🤖 AI: {content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input box
        user_input = st.text_input(
            "Chat with AI about your job search ...",
            value=st.session_state["chat_input"]
        )

        if st.button("Send"):
            if user_input.strip():
                st.session_state["chat_history"].append({"role": "user", "content": user_input})
                reply = get_response(user_input)
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()
            st.session_state["chat_input"] = ""


if __name__ == "__main__":
    main()
