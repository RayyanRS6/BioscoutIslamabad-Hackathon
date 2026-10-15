import datetime
import html
import io
import math
import os
import re
import threading
import uuid

import folium
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from streamlit_folium import st_folium

st.set_page_config(page_title="BioScout Islamabad", page_icon="🌿", layout="wide")

# --- Configuration & Setup ---
# Paths are relative to this file so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBSERVATIONS_FILE = os.path.join(BASE_DIR, 'observations.csv')
KB_DOCS_DIR = os.path.join(BASE_DIR, 'kb_docs')
UPLOADED_IMAGES_DIR = os.path.join(BASE_DIR, 'uploaded_images')
os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)
os.makedirs(KB_DOCS_DIR, exist_ok=True)

OBSERVATION_COLUMNS = ['observation_id', 'user_id', 'species_name', 'common_name',
                       'date_observed', 'location_text', 'latitude', 'longitude',
                       'image_filename', 'notes']
TEXT_COLUMNS = ['user_id', 'species_name', 'common_name', 'location_text', 'image_filename', 'notes']
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
ISLAMABAD_CENTER = [33.6844, 73.0479]

SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
IMAGE_MODEL_NAME = 'openai/clip-vit-base-patch32'

# Minimum relevance score for a passage to be used in an answer, per search method
MIN_RELEVANCE = {'semantic': 0.2, 'keyword': 0.1}

EXAMPLE_QUESTIONS = [
    "What birds can I see at Rawal Lake?",
    "Where have leopards been spotted?",
    "Which invasive species are a concern?",
]


# --- AI Models ---
# Models are loaded on first use (and downloaded on the very first run). If one can't be
# loaded, e.g. no internet connection, the app keeps working without it and tries again
# in the next browser session.
@st.cache_resource(show_spinner="Loading the text model (downloaded on first run)...")
def _load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SENTENCE_MODEL_NAME)


@st.cache_resource(show_spinner="Loading the image model (downloaded on first run)...")
def _load_image_classifier():
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-image-classification", model=IMAGE_MODEL_NAME, device=device)


def get_model(name, loader):
    error_key = f"{name}_error"
    if error_key in st.session_state:
        return None
    try:
        return loader()
    except Exception as e:
        st.session_state[error_key] = str(e)
        return None


def model_error(name):
    """Short description of why a model couldn't be loaded."""
    lines = st.session_state.get(f"{name}_error", '').strip().splitlines()
    return lines[0][:200] if lines else "unknown error"


@st.cache_resource
def get_write_lock():
    # Shared by all sessions so simultaneous submissions don't get the same ID
    return threading.Lock()


# --- Observations ---
def load_observations():
    if os.path.exists(OBSERVATIONS_FILE) and os.path.getsize(OBSERVATIONS_FILE) > 0:
        df = pd.read_csv(OBSERVATIONS_FILE).reindex(columns=OBSERVATION_COLUMNS)
    else:
        df = pd.DataFrame(columns=OBSERVATION_COLUMNS)
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna('').astype(str).str.strip()
    for col in ['observation_id', 'latitude', 'longitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date_observed'] = pd.to_datetime(df['date_observed'], format='mixed', errors='coerce')
    return df


def save_observation(data):
    """Append an observation to the CSV and return its new ID."""
    with get_write_lock():
        ids = load_observations()['observation_id'].dropna()
        observation_id = int(ids.max()) + 1 if not ids.empty else 1
        row = pd.DataFrame([{**data, 'observation_id': observation_id}], columns=OBSERVATION_COLUMNS)
        file_exists = os.path.exists(OBSERVATIONS_FILE) and os.path.getsize(OBSERVATIONS_FILE) > 0
        line_end = '\n'
        if file_exists:
            with open(OBSERVATIONS_FILE, 'rb+') as f:
                # Match the file's existing line endings and make sure the new row starts on its own line
                if f.readline().endswith(b'\r\n'):
                    line_end = '\r\n'
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b'\n':
                    f.write(line_end.encode())
        row.to_csv(OBSERVATIONS_FILE, mode='a', header=not file_exists, index=False, lineterminator=line_end)
    return observation_id


def save_uploaded_image(uploaded_file):
    stem, ext = os.path.splitext(os.path.basename(uploaded_file.name))
    safe_stem = re.sub(r'[^A-Za-z0-9_-]+', '_', stem).strip('_')[:60] or 'image'
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    image_filename = f"{timestamp}_{uuid.uuid4().hex[:6]}_{safe_stem}{ext.lower()}"
    with open(os.path.join(UPLOADED_IMAGES_DIR, image_filename), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return image_filename


def filter_observations(df, species_query, location_query):
    species_query, location_query = species_query.strip(), location_query.strip()
    if species_query:
        df = df[df['species_name'].str.contains(species_query, case=False, regex=False) |
                df['common_name'].str.contains(species_query, case=False, regex=False)]
    if location_query:
        df = df[df['location_text'].str.contains(location_query, case=False, regex=False)]
    return df


def observation_title(row):
    if row['common_name'] and row['species_name']:
        return f"{row['common_name']} ({row['species_name']})"
    return row['common_name'] or row['species_name'] or "Unknown species"


def format_date(value):
    return value.strftime('%Y-%m-%d') if pd.notna(value) else "unknown date"


def has_coordinates(row):
    return pd.notna(row['latitude']) and pd.notna(row['longitude'])


def map_view(located):
    """Center and zoom level that fit all markers on a ~350px map.

    Computed here instead of with Map.fit_bounds, which zooms all the way in when the map
    is drawn inside a tab that isn't visible yet.
    """
    if located.empty:
        return ISLAMABAD_CENTER, 11
    center_lat = (located['latitude'].min() + located['latitude'].max()) / 2
    center_lon = (located['longitude'].min() + located['longitude'].max()) / 2
    # Latitude degrees are stretched by 1/cos(latitude) on the Web Mercator map
    lat_span = (located['latitude'].max() - located['latitude'].min()) / math.cos(math.radians(center_lat))
    lon_span = located['longitude'].max() - located['longitude'].min()
    # At zoom 0 the 360° world is 256px wide, and each zoom level doubles that
    zoom = int(math.log2(360 * 350 / 256 / max(lat_span, lon_span, 0.01)))
    return [center_lat, center_lon], max(2, min(zoom, 14))


def escape_for_map(text):
    # Folium puts popup and tooltip HTML inside JavaScript template literals (`...${}...`),
    # so user-submitted text needs those characters escaped as well as HTML
    return html.escape(text).replace('`', '&#96;').replace('$', '&#36;')


def build_observation_map(df):
    located = df[df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180)]
    center, zoom = map_view(located)
    m = folium.Map(location=center, zoom_start=zoom)
    for _, row in located.iterrows():
        title = escape_for_map(observation_title(row))
        details = escape_for_map(f"{row['location_text']} · {format_date(row['date_observed'])}")
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=folium.Popup(f"<b>{title}</b><br>{details}", max_width=250),
            tooltip=title,
        ).add_to(m)
    return m, len(located)


# --- Knowledge Base ---
SPECIES_LINE = re.compile(r'^\s*[-*]\s+(?P<common>[^()\n]+?)\s*\((?P<scientific>[A-Z][a-z]+ [a-z]+)\)', re.MULTILINE)


def load_kb_documents():
    docs = []
    for filename in sorted(os.listdir(KB_DOCS_DIR)):
        if filename.endswith((".txt", ".md")):
            with open(os.path.join(KB_DOCS_DIR, filename), 'r', encoding='utf-8') as f:
                docs.append({"text": f.read(), "source": filename})
    return docs


def split_into_sections(doc):
    """Split a markdown document into one passage per section, titled with its heading path."""
    passages, headings, body = [], [], []

    def add_passage():
        content = '\n'.join(body).strip()
        # Skip empty sections and one-line lead-ins like "The park hosts a variety of wildlife:"
        if content and not ('\n' not in content and content.endswith(':')):
            title = ' › '.join(heading for _, heading in headings) or doc['source']
            passages.append({'kind': 'kb', 'title': title, 'text': content, 'source': doc['source']})
        body.clear()

    for line in doc['text'].splitlines():
        match = re.match(r'^(#{1,6})\s+(.+)', line)
        if match:
            add_passage()
            level = len(match.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, match.group(2).strip())]
        else:
            body.append(line)
    add_passage()
    return passages


def observation_passages(df):
    passages = []
    for _, row in df.iterrows():
        location = row['location_text'] or "an unknown location"
        text = f"{observation_title(row)} seen at {location} on {format_date(row['date_observed'])}."
        if row['notes']:
            text += f" Notes: {row['notes']}"
        observation_ref = f"#{int(row['observation_id'])}" if pd.notna(row['observation_id']) else ""
        passages.append({'kind': 'observation', 'title': observation_title(row),
                         'text': ' '.join(text.split()),
                         'source': f"Observation {observation_ref} by {row['user_id'] or 'anonymous'}"})
    return passages


def species_catalogue():
    """Species known around Islamabad, as {common name: scientific name}, from the knowledge base and observations."""
    catalogue = {}
    seen_scientific = set()

    def add(common, scientific):
        common, scientific = ' '.join(common.split()), ' '.join(scientific.split())
        if common and scientific and scientific.lower() not in seen_scientific and common not in catalogue:
            catalogue[common] = scientific
            seen_scientific.add(scientific.lower())

    for doc in load_kb_documents():
        for match in SPECIES_LINE.finditer(doc['text']):
            add(match.group('common'), match.group('scientific'))
    for _, row in load_observations().iterrows():
        add(row['common_name'], row['species_name'])
    return catalogue


@st.cache_data(show_spinner=False, max_entries=8)
def embed_texts(texts):
    return _load_sentence_model().encode(list(texts), normalize_embeddings=True)


def keyword_tokens(text):
    words = [w for w in re.findall(r'[a-z]+', text.lower()) if w not in ENGLISH_STOP_WORDS]
    # Crude plural folding so "leopards" matches "leopard"
    return [w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w for w in words]


def keyword_scores(query, texts):
    vectorizer = TfidfVectorizer(tokenizer=keyword_tokens, token_pattern=None, lowercase=False)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:  # none of the passages contain any usable words
        return np.zeros(len(texts))
    return (matrix @ vectorizer.transform([query]).T).toarray().ravel()


def rank_passages(query, passages):
    """Score every passage against the query. Returns ([(passage, score)] best first, search method)."""
    texts = [f"{p['title']}\n{p['text']}" for p in passages]
    scores = keyword_scores(query, texts)
    method = 'keyword'
    model = get_model('sentence_model', _load_sentence_model)
    if model is not None:
        semantic_scores = embed_texts(tuple(texts)) @ model.encode([query], normalize_embeddings=True)[0]
        # Blend in keyword matches so specific names ("snakes", "Lantana") aren't drowned out by place names
        scores = 0.7 * semantic_scores + 0.3 * scores
        method = 'semantic'
    order = np.argsort(scores)[::-1]
    return [(passages[i], float(scores[i])) for i in order], method


def top_matches(ranked, kind, min_score, limit):
    matches = [(p, s) for p, s in ranked if p['kind'] == kind and s >= min_score]
    if matches:
        # Keep only results reasonably close to the best one
        best = matches[0][1]
        matches = [(p, s) for p, s in matches if s >= 0.75 * best]
    return matches[:limit]


def answer_question(query):
    passages = [p for doc in load_kb_documents() for p in split_into_sections(doc)]
    passages += observation_passages(load_observations())
    if not passages:
        return [], [], None
    ranked, method = rank_passages(query, passages)
    min_score = MIN_RELEVANCE[method]
    return top_matches(ranked, 'kb', min_score, 3), top_matches(ranked, 'observation', min_score, 5), method


# --- Streamlit App UI ---
def apply_species_suggestion(form_id, common_name, species_name):
    st.session_state[f'common_name_{form_id}'] = common_name
    st.session_state[f'species_name_{form_id}'] = species_name


def set_question(question):
    st.session_state['question'] = question


@st.cache_data(show_spinner="AI analyzing image...", max_entries=32)
def identify_species(image_bytes, candidate_labels, top_k=3):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    predictions = _load_image_classifier()(image, candidate_labels=list(candidate_labels),
                                           hypothesis_template="a photo of a {}.")
    return [(p['label'], float(p['score'])) for p in predictions[:top_k]]


def show_species_suggestions(image_bytes, form_id):
    st.subheader("AI Species Suggestions")
    catalogue = species_catalogue()
    if not catalogue:
        st.info("Add species to the knowledge base to enable AI suggestions.")
        return
    if get_model('image_classifier', _load_image_classifier) is None:
        st.info("AI species identification isn't available right now (the image model couldn't be loaded). "
                "You can still submit your observation.")
        st.caption(f"Reason: {model_error('image_classifier')}")
        return
    try:
        suggestions = identify_species(image_bytes, tuple(sorted(catalogue)))
    except Exception as e:
        st.warning(f"Couldn't analyze this image: {e}")
        return
    st.caption("Closest matches among species recorded around Islamabad. Please verify before using one.")
    for i, (common_name, score) in enumerate(suggestions):
        species_name = catalogue[common_name]
        col_name, col_button = st.columns([3, 1], vertical_alignment="center")
        col_name.markdown(f"**{common_name}** (*{species_name}*) · {score:.0%} match")
        col_button.button("Use this", key=f"use_suggestion_{form_id}_{i}",
                          on_click=apply_species_suggestion, args=(form_id, common_name, species_name))


st.title("🌿 BioScout Islamabad 🦉")
st.caption("AI for Community Biodiversity & Sustainable Insights")

# Sidebar for community stats
with st.sidebar:
    st.header("BioScout Community Stats")
    df = load_observations()
    if not df.empty:
        species = df['species_name'].where(df['species_name'] != '', df['common_name']).str.lower()
        col_obs, col_species = st.columns(2)
        col_obs.metric("Observations", len(df))
        col_species.metric("Species", species[species != ''].nunique())
        st.write("**Top Observers:**")
        for user, count in df['user_id'].replace('', 'anonymous').value_counts().head(3).items():
            st.write(f"👤 {user}: {count} observation{'s' if count != 1 else ''}")
    else:
        st.write("No observations yet. Be the first to contribute!")

tab1, tab2, tab3 = st.tabs(["Submit Observation", "View Observations", "Ask BioScout AI"])

# --- TAB 1: Submit Observation ---
with tab1:
    st.header("Submit a Biodiversity Observation")
    # Bumping form_id after a submission gives the form fresh, empty widgets
    form_id = st.session_state.setdefault('form_id', 0)
    if 'submit_message' in st.session_state:
        st.success(st.session_state.pop('submit_message'))

    uploaded_image = st.file_uploader("Upload Image of Species (optional)", type=IMAGE_TYPES,
                                      key=f"image_{form_id}")
    image_is_valid = False
    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        try:
            Image.open(io.BytesIO(image_bytes)).load()
            image_is_valid = True
        except Exception:
            st.error("That file couldn't be read as an image. Please upload a JPG, PNG or WebP photo.")
        if image_is_valid:
            col_image, col_ai = st.columns([1, 2])
            col_image.image(image_bytes, caption="Uploaded Image", width=250)
            with col_ai:
                show_species_suggestions(image_bytes, form_id)

    st.session_state.setdefault('user_id', 'anonymous_hiker')
    today = datetime.date.today()
    with st.form("observation_form"):
        user_id = st.text_input("Your User ID/Name", key='user_id')
        col_common, col_species = st.columns(2)
        common_name = col_common.text_input("Common Name", placeholder="e.g., Common Leopard",
                                            key=f"common_name_{form_id}")
        species_name = col_species.text_input("Species Name (Scientific)", placeholder="e.g., Panthera pardus",
                                              key=f"species_name_{form_id}")
        col_date, col_location = st.columns(2)
        date_observed = col_date.date_input("Date of Observation", value=today, max_value=today,
                                            key=f"date_{form_id}")
        location_text = col_location.text_input("Location", placeholder="e.g., Trail 5, Margalla Hills",
                                                key=f"location_{form_id}")

        col_lat, col_lon = st.columns(2)
        latitude = col_lat.number_input("Latitude (optional)", min_value=-90.0, max_value=90.0, value=None,
                                        format="%.6f", placeholder="33.748000", key=f"latitude_{form_id}")
        longitude = col_lon.number_input("Longitude (optional)", min_value=-180.0, max_value=180.0, value=None,
                                         format="%.6f", placeholder="73.060000", key=f"longitude_{form_id}")

        notes = st.text_area("Additional Notes (behavior, habitat, etc.)", key=f"notes_{form_id}")

        submitted = st.form_submit_button("Submit Observation", type="primary")

    if submitted:
        errors = []
        if not common_name.strip() and not species_name.strip():
            errors.append("Please provide at least a common or scientific name.")
        if not location_text.strip():
            errors.append("Please provide the location of the observation.")
        if (latitude is None) != (longitude is None):
            errors.append("Please provide both latitude and longitude, or leave both empty.")
        if uploaded_image is not None and not image_is_valid:
            errors.append("Please remove or replace the image that couldn't be read.")
        for error in errors:
            st.error(error)

        if not errors:
            observation_id = save_observation({
                'user_id': user_id.strip() or 'anonymous',
                'species_name': ' '.join(species_name.split()),
                'common_name': ' '.join(common_name.split()),
                'date_observed': date_observed.isoformat(),
                'location_text': location_text.strip(),
                'latitude': latitude,
                'longitude': longitude,
                'image_filename': save_uploaded_image(uploaded_image) if uploaded_image is not None else '',
                'notes': notes.strip(),
            })
            st.session_state['form_id'] = form_id + 1
            st.session_state['submit_message'] = f"Observation #{observation_id} submitted successfully! Thank you for contributing."
            st.rerun()

# --- TAB 2: View Observations ---
with tab2:
    st.header("View Biodiversity Observations")

    df = load_observations()
    if not df.empty:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            species_filter = st.text_input("Filter by species name")
        with col2:
            location_filter = st.text_input("Filter by location")

        df = filter_observations(df, species_filter, location_filter)
        df = df.sort_values('date_observed', ascending=False, na_position='last')
        st.caption(f"Showing {len(df)} observation{'s' if len(df) != 1 else ''}")

        # Map view
        st.subheader("Observation Map")
        observation_map, located_count = build_observation_map(df)
        if located_count:
            st_folium(observation_map, key="observation_map", height=450, use_container_width=True,
                      returned_objects=[])
        else:
            st.info("None of these observations have coordinates to show on the map.")

        # Display observations
        st.subheader("Observations")
        if df.empty:
            st.info("No observations match your filters.")
        for _, row in df.iterrows():
            with st.expander(f"{observation_title(row)} - {format_date(row['date_observed'])}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    image_path = os.path.join(UPLOADED_IMAGES_DIR, os.path.basename(row['image_filename']))
                    if row['image_filename'] and os.path.isfile(image_path):
                        try:
                            st.image(image_path, width=200)
                        except Exception:
                            st.caption("📷 Photo couldn't be displayed")
                    else:
                        st.caption("📷 No photo available")
                with col2:
                    st.markdown(f"**Location:** {row['location_text'] or 'Not specified'}")
                    if has_coordinates(row):
                        st.markdown(f"**Coordinates:** {row['latitude']:.5f}, {row['longitude']:.5f}")
                    if row['notes']:
                        st.markdown(f"**Notes:** {row['notes']}")
                    st.markdown(f"**Observer:** {row['user_id'] or 'anonymous'}")
    else:
        st.info("No observations submitted yet. Be the first to contribute!")

# --- TAB 3: Ask BioScout AI ---
with tab3:
    st.header("Ask BioScout AI")
    st.write("Ask questions about biodiversity in Islamabad and surrounding areas. "
             "Answers come from the BioScout knowledge base and community observations.")

    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(example_cols, EXAMPLE_QUESTIONS):
        col.button(example, on_click=set_question, args=(example,))

    query = st.text_input("Your question", placeholder="e.g., What birds are common in Margalla Hills?",
                          key='question')

    if query.strip():
        with st.spinner("Searching knowledge base..."):
            kb_matches, observation_matches, method = answer_question(query.strip())

        st.subheader("Answer")
        if not kb_matches and not observation_matches:
            st.info("I couldn't find anything about that in the knowledge base or community observations. "
                    "Try asking about local species, Margalla Hills, Rawal Lake or urban wildlife.")
        if kb_matches:
            st.markdown("**From the knowledge base:**")
            for passage, _ in kb_matches:
                with st.container(border=True):
                    st.markdown(f"**{passage['title']}**")
                    st.markdown(passage['text'])
                    st.caption(f"Source: {passage['source']}")
        if observation_matches:
            st.markdown("**Related community observations:**")
            for passage, _ in observation_matches:
                st.markdown(f"- {passage['text']} _({passage['source']})_")

        if method:
            with st.expander("How this answer was found"):
                if method == 'semantic':
                    st.write(f"Semantic search with the `{SENTENCE_MODEL_NAME}` sentence model, combined with keyword matching.")
                else:
                    st.write("Keyword search only (the semantic search model couldn't be loaded: "
                             f"{model_error('sentence_model')}).")
                for passage, score in kb_matches + observation_matches:
                    st.write(f"- {passage['title']} ({passage['source']}): relevance {score:.2f}")
