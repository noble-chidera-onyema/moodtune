"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MoodTune – Persuasive AI Music Advisor                                      ║
║  Author  : Noble Chidera Onyema                                              ║
║  Module  : CMP512 – Case Studies and Research Methods                        ║
║  Degree  : MSc Applied Artificial Intelligence & User Experience             ║
║  Repo    : https://github.com/noble-chidera-onyema/moodtune                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  APPLIED AI COMPONENTS                                                        ║
║  1. LLM (Groq / LLaMA-3.3-70B) for natural language mood understanding       ║
║     and music recommendation generation.                                      ║
║  2. Spotify Audio Features API – machine-extracted acoustic properties        ║
║     (valence, energy, danceability, tempo) used to validate LLM              ║
║     recommendations against the user's detected mood profile.                 ║
║  3. Mood taxonomy (12 profiles) maps natural language input to               ║
║     quantitative audio feature targets, creating a rule-based AI             ║
║     validation layer on top of the generative LLM.                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  See README.md for full UX rationale, persuasion theory grounding,           ║
║  ethics framework, and academic references.                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
from openai import OpenAI
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os, re, random, urllib.parse, hashlib
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Environment validation — fail clearly if .env is missing/incomplete ──────
_required_keys = ["GROQ_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"]
_missing = [k for k in _required_keys if not os.getenv(k)]

st.set_page_config(
    page_title="MoodTune – Music Recommendation Study",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if _missing:
    st.error(
        f"❌ Missing environment variables: {', '.join(_missing)}.\n\n"
        f"Copy `.env.example` to `.env` and add your own API keys before running."
    )
    st.stop()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header  { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 840px; }

h1 { font-family: 'DM Serif Display', serif !important; font-size: 2rem !important; }
h2, h3 { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }

.progress-label {
    font-size: 11px; color: #9ca3af; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 6px;
}
.progress-bar  { display: flex; gap: 6px; margin-bottom: 1.8rem; }
.progress-step { flex: 1; height: 4px; border-radius: 2px; background: rgba(139,92,246,0.18); }
.progress-step.done   { background: #6d28d9; }
.progress-step.active { background: #8b5cf6; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#7c3aed,#8b5cf6) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 500 !important; padding: 0.55rem 1.4rem !important;
    transition: opacity .2s;
}
.stButton > button[kind="primary"]:hover { opacity: .88; }

button[data-testid="baseButton-secondary"][kind="secondary"]:has-text("Withdraw"),
.stButton > button:is([data-testid*="withdraw"], [key*="withdraw"]) {
    background: rgba(220,38,38,.12) !important;
    border: 1.5px solid rgba(220,38,38,.6) !important;
    color: #f87171 !important;
}

[data-testid="stChatMessage"] { border-radius: 14px !important; padding: .25rem !important; }
[data-testid="stExpander"]    { border: 1px solid rgba(139,92,246,.2) !important; border-radius: 12px !important; }
[data-testid="stAlert"]       { border-radius: 12px !important; border-left: 4px solid #8b5cf6 !important; }
hr { border-color: rgba(139,92,246,.15) !important; }

.withdraw-float {
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
}
.withdraw-btn {
    background: rgba(30,10,10,0.92) !important;
    border: 1px solid rgba(220,38,38,0.6) !important;
    color: #f87171 !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-size: 13px !important; font-weight: 600 !important;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(220,38,38,0.25);
    transition: all .2s;
    font-family: "DM Sans", sans-serif;
}
.withdraw-btn:hover {
    background: rgba(220,38,38,0.3) !important;
    box-shadow: 0 4px 24px rgba(220,38,38,0.45);
}
.withdraw-confirm-box {
    position: fixed; bottom: 72px; right: 24px; z-index: 9999;
    background: rgba(20,5,5,0.97);
    border: 1px solid rgba(220,38,38,0.5);
    border-radius: 14px; padding: 16px 20px; width: 260px;
    box-shadow: 0 8px 32px rgba(220,38,38,0.2);
}

button[kind="secondary"] {
    background: rgba(220,38,38,.10) !important;
    border: 1.5px solid rgba(220,38,38,.55) !important;
    color: #f87171 !important;
    border-radius: 8px !important;
}
button[kind="secondary"]:hover {
    background: rgba(220,38,38,.22) !important;
}

[data-testid="stSidebar"] {
    min-width: 180px !important; max-width: 180px !important;
    background: rgba(15,3,3,0.97) !important;
    border-right: 1px solid rgba(220,38,38,0.25) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(220,38,38,0.12) !important;
    border: 1.5px solid rgba(220,38,38,0.6) !important;
    color: #f87171 !important;
    border-radius: 10px !important;
    font-weight: 600 !important; font-size: 13px !important;
    width: 100% !important; padding: 10px 0 !important;
    margin-top: 8px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(220,38,38,0.28) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    font-size: 12px !important; padding: 6px 0 !important;
}

.feat-bar-bg {
    background: rgba(139,92,246,.15); border-radius: 4px;
    height: 6px; width: 100%; margin: 2px 0 6px;
}
.feat-bar-fill {
    background: linear-gradient(90deg,#7c3aed,#a78bfa);
    border-radius: 4px; height: 6px;
}
.feat-label { font-size: 11px; color: #9ca3af; margin: 0; }
.feat-value { font-size: 11px; color: #c4b5fd; font-weight: 500; float: right; }
</style>
""", unsafe_allow_html=True)


# ── Clients ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_clients():
    groq = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    try:
        spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
        ))
        spotify.search(q="test", type="track", limit=1)
        return groq, spotify
    except Exception:
        return groq, None

try:
    client, sp = get_clients()
except Exception as e:
    st.error(f"❌ Client init failed. Check your .env keys. Error: {e}")
    st.stop()

CSV_PATH         = "test_results.csv"
RESEARCHER_EMAIL = os.getenv("RESEARCHER_EMAIL", "researcher@example.com")

# ── Pictorial Mood Induction Images (Unsplash – free academic use licence) ────
INDUCTION_IMAGES = [
    "https://images.unsplash.com/photo-1448375240586-882707db888b?w=600&q=70&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=70&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1476820865390-c52aeebb9891?w=600&q=70&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1504280390367-361c6d9f38f4?w=600&q=70&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=600&q=70&auto=format&fit=crop",
]

INDUCTION_IMAGE_ALTS = [
    "A quiet misty forest path",
    "A still lake with soft overcast light",
    "Bare winter trees in grey light",
    "A foggy empty road disappearing into the distance",
    "Soft golden morning light filtering through trees",
]

CONSENT_IMAGE = "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=900&q=70&auto=format&fit=crop"

# ── Ethnicity (UK Census 2021) ────────────────────────────────────────────────
ETHNICITY_OPTIONS = [
    "Prefer not to say",
    "White – British", "White – Irish", "White – Other",
    "Mixed – White and Black Caribbean", "Mixed – White and Black African",
    "Mixed – White and Asian", "Mixed – Other",
    "Asian – Indian", "Asian – Pakistani", "Asian – Bangladeshi",
    "Asian – Chinese", "Asian – Other",
    "Black – African", "Black – Caribbean", "Black – Other",
    "Arab", "Any other ethnic group",
]

# ── Mood taxonomy → quantitative audio feature targets ────────────────────────
MOOD_PROFILES = {
    "sad":        {"desc_valence":"uplifting or gently comforting","desc_energy":"low-to-medium","avoid":"melancholic or hopeless lyrics",           "target_valence":(0.4,1.0),"target_energy":(0.2,0.7)},
    "anxious":    {"desc_valence":"calming and reassuring",        "desc_energy":"low",          "avoid":"fast tempo, aggressive or distressing",     "target_valence":(0.3,0.8),"target_energy":(0.1,0.5)},
    "stressed":   {"desc_valence":"soothing and positive",         "desc_energy":"low-to-medium","avoid":"chaotic or lyrically negative songs",        "target_valence":(0.3,0.8),"target_energy":(0.1,0.6)},
    "angry":      {"desc_valence":"calming or empowering",         "desc_energy":"medium",       "avoid":"songs glorifying violence or hatred",        "target_valence":(0.3,0.9),"target_energy":(0.3,0.7)},
    "happy":      {"desc_valence":"joyful and celebratory",        "desc_energy":"medium-high",  "avoid":"depressing or lyrically dark content",      "target_valence":(0.6,1.0),"target_energy":(0.5,1.0)},
    "focused":    {"desc_valence":"neutral and rhythmically steady","desc_energy":"medium",       "avoid":"distracting lyrics or chaotic structure",   "target_valence":(0.3,0.7),"target_energy":(0.3,0.7)},
    "tired":      {"desc_valence":"gentle and uplifting",          "desc_energy":"low-to-medium","avoid":"aggressive or overly loud tracks",           "target_valence":(0.4,0.8),"target_energy":(0.2,0.6)},
    "lonely":     {"desc_valence":"warm and connecting",           "desc_energy":"low-to-medium","avoid":"songs that deepen feelings of isolation",    "target_valence":(0.4,0.8),"target_energy":(0.2,0.6)},
    "excited":    {"desc_valence":"upbeat and celebratory",        "desc_energy":"high",         "avoid":"slow or melancholic content",               "target_valence":(0.6,1.0),"target_energy":(0.6,1.0)},
    "bored":      {"desc_valence":"fun and engaging",              "desc_energy":"medium-high",  "avoid":"repetitive or predictable tracks",          "target_valence":(0.5,1.0),"target_energy":(0.4,0.9)},
    "nervous":    {"desc_valence":"calming and confidence-boosting","desc_energy":"low-to-medium","avoid":"chaotic or anxiety-inducing content",       "target_valence":(0.4,0.8),"target_energy":(0.2,0.6)},
    "overwhelmed":{"desc_valence":"soothing and grounding",        "desc_energy":"low",          "avoid":"busy or lyrically dense songs",             "target_valence":(0.3,0.7),"target_energy":(0.1,0.4)},
    "default":    {"desc_valence":"positive and broadly appealing","desc_energy":"medium",       "avoid":"explicit, offensive or insensitive content","target_valence":(0.4,0.8),"target_energy":(0.3,0.7)},
}

def detect_mood(text: str) -> dict:
    tl = text.lower()
    for mood, profile in MOOD_PROFILES.items():
        if mood in tl:
            return profile
    return MOOD_PROFILES["default"]


def get_audio_features(track_id: str) -> dict | None:
    if sp is None:
        return None
    try:
        feats = sp.audio_features([track_id])
        if feats and feats[0]:
            return feats[0]
    except Exception:
        pass
    return None


def mood_match_score(features: dict, mood_profile: dict) -> tuple[float, str]:
    v_min, v_max = mood_profile["target_valence"]
    e_min, e_max = mood_profile["target_energy"]
    valence = features.get("valence", 0.5)
    energy  = features.get("energy",  0.5)
    v_score = 100 if v_min <= valence <= v_max else max(0, 100 - abs(valence - (v_min+v_max)/2) * 200)
    e_score = 100 if e_min <= energy  <= e_max else max(0, 100 - abs(energy  - (e_min+e_max)/2) * 200)
    score   = round((v_score + e_score) / 2)
    if score >= 75:   verdict = "✅ Good mood match"
    elif score >= 50: verdict = "⚠️ Partial mood match"
    else:             verdict = "❌ Low mood match"
    return score, verdict


def render_audio_features(features: dict, mood_profile: dict):
    score, verdict = mood_match_score(features, mood_profile)
    v   = features.get("valence",      0)
    e   = features.get("energy",       0)
    d   = features.get("danceability", 0)
    tmp = features.get("tempo",        0)
    st.markdown(
        f"""
        <div style="background:rgba(139,92,246,.07);border:1px solid rgba(139,92,246,.2);
                    border-radius:10px;padding:10px 14px;margin:8px 0;">
          <div style="font-size:12px;font-weight:600;color:#c4b5fd;margin-bottom:8px;">
            🎛 Audio Analysis &nbsp;·&nbsp; {verdict} ({score}/100)
          </div>
          <p style="font-size:10px;color:#6b7280;margin:0 0 8px;">
            <strong>Valence</strong> = musical positivity (high = happy, low = sad) &nbsp;·&nbsp;
            <strong>Energy</strong> = intensity (high = powerful, low = gentle) &nbsp;·&nbsp;
            <strong>Danceability</strong> = rhythmic suitability for dancing
          </p>
          <p class="feat-label">Valence (positivity) <span class="feat-value">{v:.0%}</span></p>
          <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{v*100:.0f}%"></div></div>
          <p class="feat-label">Energy <span class="feat-value">{e:.0%}</span></p>
          <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{e*100:.0f}%"></div></div>
          <p class="feat-label">Danceability <span class="feat-value">{d:.0%}</span></p>
          <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{d*100:.0f}%"></div></div>
          <p class="feat-label">Tempo <span class="feat-value">{tmp:.0f} BPM</span></p>
          <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{min(tmp/200,1)*100:.0f}%"></div></div>
          <p style="font-size:10px;color:#6b7280;margin-top:6px;margin-bottom:0;">
            Audio features powered by Spotify's audio analysis models.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


SAFETY_RULES = """
CONTENT SAFETY – mandatory for every recommendation:
1. NO explicit lyrics (graphic sex, violence, drug glorification).
2. NO songs associated with self-harm, suicide, or hopelessness.
3. NO racist, homophobic, misogynistic, or discriminatory content.
4. PREFER widely recognised mainstream songs suitable for university students aged 18–24.
5. NEVER recommend the same artist twice in one response.
6. NEVER mention in your reply that you are replacing, substituting, or filtering a song.
   If a song does not meet safety guidelines, silently choose a different one instead.
   Do NOT write phrases like "X is not suitable, I will replace it with Y".
   Just recommend Y directly without any explanation of the replacement.
7. If the user expresses severe distress (self-harm or suicidal ideation), do NOT recommend
   music. Respond ONLY with:
   "I can hear that you're going through something really difficult right now.
   Please speak to someone you trust, or contact your university wellbeing service."
"""

def build_history_context(messages: list[dict]) -> str:
    prev = []
    for m in messages:
        if m["role"] == "assistant" and m.get("songs"):
            for s in m["songs"]:
                prev.append(f"{s['title']} by {s['artist']}")
    if not prev:
        return ""
    return (
        f"\nPREVIOUSLY RECOMMENDED – do NOT suggest these again: {'; '.join(prev)}. "
        "Choose completely different songs AND different artists from any prior recommendation."
    )


_NON_ARTIST_WORDS = {
    "a", "an", "the", "me", "my", "i", "you", "we", "it", "this", "that",
    "is", "are", "was", "be", "been", "have", "has", "do", "does", "did",
    "to", "for", "of", "in", "on", "at", "by", "from", "with", "about",
    "into", "up", "down", "low", "high", "good", "bad", "fast", "slow",
    "different", "various", "mixed", "random", "some", "all", "any", "many",
    "music", "song", "songs", "track", "tracks", "genre", "genres", "type",
    "artist", "artists", "band", "bands", "playlist", "list", "mix",
    "please", "need", "want", "give", "find", "play", "suggest", "recommend",
    "feeling", "feel", "mood", "today", "now", "just", "really", "very",
    "lift", "boost", "cheer", "calm", "help", "make", "let", "like",
    "note", "since", "only", "but", "and", "or", "nor", "so", "yet",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "new", "old", "big", "cool", "nice", "great", "best", "top",
}

_GENRE_KEYWORDS = [
    "afrobeat", "afropop", "afro", "afrofusion", "afroswing",
    "hip hop", "hip-hop", "hiphop", "rap", "trap", "drill",
    "r&b", "rnb", "soul", "funk", "blues", "jazz", "neo soul",
    "pop", "dance pop", "indie pop", "synth pop",
    "rock", "indie", "alternative", "metal", "punk", "grunge",
    "classical", "orchestra", "piano", "acoustic",
    "country", "folk", "bluegrass",
    "reggae", "dancehall", "ska",
    "electronic", "edm", "house", "techno", "trance", "dubstep",
    "lo-fi", "lofi", "lo fi", "chillout", "ambient",
    "gospel", "worship", "christian",
    "highlife", "fuji", "juju", "apala",
    "amapiano", "gqom", "kwaito",
    "kizomba", "zouk", "kuduro",
    "bongo", "bongo flava", "gengetone", "benga",
    "afrobeats", "naija", "naija pop",
    "nigerian rap", "nigerian hip hop", "naija rap", "afrorap",
    "afro hiphop", "afro hip hop", "african rap", "african hip hop",
    "nollywood", "street pop", "alté", "alte",
    "latin", "salsa", "reggaeton", "bachata", "cumbia",
    "k-pop", "j-pop", "bollywood", "mandopop", "afroswing",
    "arabic", "arabic pop", "arabic music", "oud music", "khaleeji",
    "arabic hip hop", "arabic rap", "levantine", "middle eastern",
    "egyptian pop", "arabic r&b",
    "nordic", "scandinavian", "nordic folk", "scandinavian folk",
    "norse", "viking folk", "nordic electronic", "scandinavian electronic",
    "nordic pop", "swedish pop", "norwegian folk",
    "c-pop", "cpop", "cantopop", "taiwanese pop",
    "j-rock", "j-pop", "japanese rock", "anime music",
    "afro fusion", "afroswing", "soca", "calypso", "bhangra",
    "qawwali", "sufi", "gnawa", "mbalax",
]

def extract_user_preferences(text: str) -> dict:
    text_lower = text.lower()
    count_match = re.search(
        r'\b(\d+|ten|five|three|four|six|seven|eight|nine|twenty)'
        r'\s+(?:more\s+)?(?:different\s+)?(?:songs?|tracks?|genres?|types?|kinds?|more)\b'
        r'|\bgive\s+(?:me\s+)?(\d+|ten|five|three|four|six|seven|eight|nine|twenty)\b',
        text_lower)
    requested_count = None
    if count_match:
        word = count_match.group(1) or count_match.group(2)
        word_map = {"three":3,"four":4,"five":5,"six":6,"seven":7,
                    "eight":8,"nine":9,"ten":10,"twenty":20}
        try:
            requested_count = int(word)
        except ValueError:
            requested_count = word_map.get(word, None)

    sorted_genres = sorted(_GENRE_KEYWORDS, key=len, reverse=True)
    detected_genres = []
    for g in sorted_genres:
        if re.search(r'\b' + re.escape(g) + r'\b', text_lower):
            detected_genres.append(g)
            if len(detected_genres) >= 3:
                break

    nat_match = re.search(
        r'\b(nigerian|ghanaian|kenyan|south african|tanzanian|ugandan|senegalese'
        r'|jamaican|british|american|french|spanish|korean|japanese|indian|latin'
        r'|arabic|arab|middle eastern|egyptian|lebanese|saudi|scandinavian'
        r'|nordic|norwegian|swedish|danish|finnish|chinese|mandarin|cantonese'
        r'|taiwanese|bollywood|hindi|punjabi|bhangra)\s+'
        r'(rap|hip.?hop|pop|afrobeats?|drill|trap|r&b|rnb|jazz|reggae|gospel'
        r'|folk|electronic|music|songs?)\b',
        text_lower
    )
    if nat_match:
        combo = nat_match.group(0)
        if combo not in detected_genres:
            detected_genres.insert(0, combo)

    detected_genre = detected_genres[0] if detected_genres else None
    multi_genre = len(detected_genres) > 1 or bool(re.search(
        r'\b(different|various|multiple|mixed)\s+(genres?|styles?|types?)\b', text_lower
    ))

    artists = []
    explicit_patterns = [
        r'\b(?:by|from)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})',
        r'\bartist(?:s)?\s+(?:like\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})',
    ]
    for pat in explicit_patterns:
        for m in re.finditer(pat, text):
            name = m.group(1).strip()
            if name.lower() not in _NON_ARTIST_WORDS and len(name) > 1:
                if name not in artists:
                    artists.append(name)

    paired = re.findall(
        r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)\s+(?:or|and)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)\b',
        text
    )
    for a, b in paired:
        for name in (a, b):
            if (name.lower() not in _NON_ARTIST_WORDS
                    and len(name) > 1 and name not in artists):
                artists.append(name)

    return {
        "artists":         artists[:6],
        "genre":           detected_genre,
        "multi_genre":     multi_genre,
        "requested_count": requested_count,
    }


def build_system_prompt(mode: str, mood_profile: dict, messages: list[dict],
                        user_prefs: dict | None = None) -> str:
    variety = build_history_context(messages)

    mood_instruction = (
        f"The user's mood requires music that is: {mood_profile['desc_valence']}, "
        f"with {mood_profile['desc_energy']} energy. "
        f"Avoid: {mood_profile['avoid']}."
    )

    preference_rule = ""
    if user_prefs:
        parts = []
        n = user_prefs.get("requested_count") or 2
        n = min(n, 10)

        if user_prefs["artists"]:
            artist_list = ", ".join(user_prefs["artists"])
            parts.append(
                f"MANDATORY ARTIST REQUIREMENT: The user explicitly requested songs by "
                f"{artist_list}. You MUST use ONLY these artists. "
                f"Do NOT substitute other artists under any circumstances. "
                f"Provide one song per artist listed."
            )

        if user_prefs.get("multi_genre") and not user_prefs["artists"]:
            parts.append(
                f"MULTI-GENRE REQUIREMENT: The user asked for {n} songs across DIFFERENT genres. "
                f"Each song MUST be from a COMPLETELY different genre family — "
                f"not just sub-genres of the same family. "
                f"FORBIDDEN: Pop + Indie Pop (same family), Hip-Hop + Trap (same family), "
                f"Classical + Orchestra (same family). "
                f"USE INSTEAD: genres from entirely different families, e.g. "
                f"Afrobeat, Hip-Hop, Jazz, Pop, R&B, Reggae, Classical, EDM, Country, Folk, Rock, Soul. "
                f"Each genre must be clearly and distinctly different from all others in the list."
            )
        elif user_prefs["genre"]:
            genre = user_prefs["genre"]
            genre_hints = {
                "nigerian rap":     "Olamide, Reminisce, Vector, Phyno, Ycee, Lil Kesh, Zlatan, Blaqbonez, M.I Abaga, Ice Prince",
                "naija rap":        "Olamide, Reminisce, Vector, Phyno, Ycee, Lil Kesh, Zlatan, Blaqbonez",
                "nigerian hip hop": "Olamide, Reminisce, Vector, Phyno, M.I Abaga, Ice Prince, Jesse Jagz, Blaqbonez",
                "african rap":      "Olamide, Vector, M.I Abaga, Phyno, Cassper Nyovest, AKA, Sarkodie, Nasty C",
                "african hip hop":  "Olamide, Vector, M.I Abaga, Sarkodie, Cassper Nyovest, AKA, Nasty C",
                "amapiano":         "Kabza De Small, DJ Maphorisa, Uncle Waffles, Focalistic, Mas Musiq, Sun-El Musician",
                "afrobeats":        "Wizkid, Burna Boy, Davido, Tiwa Savage, Ckay, Rema, Omah Lay, Fireboy DML, Tems",
                "afrobeat":         "Wizkid, Burna Boy, Davido, Tiwa Savage, Ckay, Rema, Omah Lay, Fireboy DML, Tems",
                "alté":             "Santi, Odunsi the Engine, Lady Donli, Cruel Santino, Amaarae",
                "alte":             "Santi, Odunsi the Engine, Lady Donli, Cruel Santino, Amaarae",
                "highlife":         "Flavour, Kcee, Phyno, Sunny Ade, Ebenezer Obey, Oriental Brothers",
                "arabic":           "Amr Diab, Fairuz, Umm Kulthum, Elissa, Nancy Ajram, Wael Kfoury, Saad Lamjarred, Hamaki",
                "arabic pop":       "Amr Diab, Nancy Ajram, Elissa, Wael Kfoury, Saad Lamjarred, Hamaki, Marwan Khoury",
                "arabic music":     "Fairuz, Umm Kulthum, Amr Diab, Abdel Halim Hafez, Wael Kfoury, Kadim Al Sahir",
                "khaleeji":         "Mohammed Abdo, Ahlam, Nabeel Shuail, Hussain Al Jassmi, Aseel Abou Bakr",
                "middle eastern":   "Amr Diab, Fairuz, Kadim Al Sahir, Wael Kfoury, Nancy Ajram, Elissa",
                "arabic hip hop":   "El Rass, Perverse, Eslam Jawaad, Narcy, Ozartizta, Stormtrap",
                "nordic":           "Aurora, Wardruna, Of Monsters and Men, Sigrid, Röyksopp, Kygo, Astrid S, Ane Brun",
                "scandinavian":     "Aurora, Röyksopp, Kygo, Sigrid, Of Monsters and Men, Astrid S, Robyn, First Aid Kit",
                "nordic folk":      "Wardruna, Heilung, Danheim, Aurora, Faun, Garmarna, Gjallarhorn",
                "scandinavian folk":"First Aid Kit, Väsen, Nordman, Gjallarhorn, Ane Brun",
                "nordic electronic":"Röyksopp, Kygo, Alan Walker, Cashmere Cat, Todd Terje",
                "norwegian folk":   "Aurora, Ane Brun, Maria Mena, Sissel, Kari Bremnes",
                "mandopop":         "Jay Chou, G.E.M., Mayday, Eason Chan, JJ Lin, Stefanie Sun, Wang Leehom, Yoga Lin",
                "c-pop":            "Jay Chou, G.E.M., Eason Chan, Mayday, JJ Lin, Jolin Tsai, TF Boys, Jackson Wang",
                "cpop":             "Jay Chou, G.E.M., Eason Chan, JJ Lin, Jolin Tsai, Mayday, Khalil Fong",
                "cantopop":         "Eason Chan, Andy Lau, Leon Lai, Jacky Cheung, Aaron Kwok, Faye Wong",
                "bollywood":        "Arijit Singh, Shreya Ghoshal, A.R. Rahman, Pritam, Shankar-Ehsaan-Loy, Badshah, Neha Kakkar, Vishal-Shekhar. IMPORTANT: Use recent tracks from 2020-2024, not just classic hits.",
                "bhangra":          "Diljit Dosanjh, Guru Randhawa, Ammy Virk, Jasmine Sandlas, Gippy Grewal",
            }
            hint = genre_hints.get(genre.lower(), "")
            artist_note = f" Key artists: {hint}." if hint else ""
            parts.append(
                f"MANDATORY GENRE REQUIREMENT: The user explicitly requested {genre} music. "
                f"ALL songs MUST be genuine {genre} — not a substitute or adjacent genre. "
                f"Do NOT replace {genre} with American or international artists unless no "
                f"{genre} artists exist.{artist_note}"
            )

        if n != 2:
            parts.append(
                f"SONG COUNT REQUIREMENT: The user asked for {n} songs. "
                f"You MUST provide exactly {n} songs numbered 1 through {n}. "
                f"Do not provide fewer."
            )

        if parts:
            preference_rule = "\n".join(parts) + "\n"
    else:
        n = 2

    n_songs = (user_prefs.get("requested_count") or 2) if user_prefs else 2
    n_songs = min(n_songs, 10)
    song_lines = ""
    for i in range(1, n_songs + 1):
        song_lines += f"{i}. 'Song Title' by Artist Name\nExplanation on a NEW LINE.\n\n"

    format_rule = (
        "FORMAT RULE – follow EXACTLY, no exceptions:\n"
        f"{song_lines}"
        "CRITICAL: Explanation MUST be on a SEPARATE LINE after the song/artist line.\n"
        "NEVER write the explanation on the same line as the artist name.\n"
        "Single quotes go around the song title ONLY – never the artist name.\n"
        "ENGAGEMENT RULE: If the user's message is very short or vague (e.g. 'give me something', 'idk', 'just music'), "
        "ask ONE friendly follow-up question first — e.g. 'What kind of mood are you in right now? '— "
        "before recommending any songs. Do not recommend generic chart music for vague prompts."
    )

    if mode == "Regular":
        return (
            "You are a neutral music recommendation assistant for university students aged 18–24. "
            "If the user writes in a language other than English, briefly acknowledge this and respond in English. "
            "Recognise culturally specific music requests even when phrased informally.\n"
            f"{mood_instruction}\n"
            f"{preference_rule}"
            f"{SAFETY_RULES}\n"
            f"{format_rule}\n"
            f"{variety}\n"
            "Tone: purely factual and informational. "
            "Match the user's energy level — if they write casually (e.g. 'Yo', 'lol'), respond in kind but remain informational. "
            "No emotional language, no encouragement, no personal comments. "
            f"Keep reply concise — about {max(80, n_songs * 35)} words total."
        )
    else:
        return (
            "You are a warm music recommendation assistant for university students aged 18–24. "
            "If the user writes partly in another language, warmly acknowledge their language and culture before responding in English. "
            "Celebrate cultural music diversity and take culturally specific requests seriously.\n"
            f"{mood_instruction}\n"
            f"{preference_rule}"
            f"{SAFETY_RULES}\n"
            f"{format_rule}\n"
            f"{variety}\n"
            "Apply these evidence-based persuasion techniques naturally:\n"
            "  (1) EMPATHY [Cialdini: Liking] – acknowledge the user's emotion warmly in 1 sentence.\n"
            "  (2) POSITIVE REINFORCEMENT [Fogg: Motivation] – affirm that music genuinely helps.\n"
            "  (3) PERSONALISATION [Fogg: Trigger] – reference exactly what the user said.\n"
            "  (4) SOCIAL PROOF [Cialdini: Social Proof] – mention something specific about why this particular song connects with people in this emotional situation (not just generic popularity claims like 'resonates with millions').\n"
            f"Keep reply concise — about {max(100, n_songs * 40)} words total."
        )


DEFAULTS = {
    "step":             0,
    "participant_id":   datetime.now().strftime("%Y%m%d_%H%M%S"),
    "assigned_mode":    None,
    "responses":        {},
    "messages":         [],
    "num_turns":        0,
    "current_mood":     "default",
    "withdrawn":        False,
    "confirm_withdraw": False,
}

_MODE_TOKENS = {
    "Regular":    hashlib.md5(b"Regular-moodtune-2026").hexdigest()[:8],
    "Persuasive": hashlib.md5(b"Persuasive-moodtune-2026").hexdigest()[:8],
}
_TOKEN_TO_MODE = {v: k for k, v in _MODE_TOKENS.items()}

def _save_to_url():
    st.query_params["s"]  = str(st.session_state.step)
    st.query_params["p"]  = st.session_state.participant_id
    mode = st.session_state.assigned_mode or ""
    st.query_params["t"]  = _MODE_TOKENS.get(mode, "")
    st.query_params["w"]  = "1" if st.session_state.withdrawn else "0"

def _restore_from_url():
    qp = st.query_params
    if "s" in qp:
        try:
            st.session_state.step = int(qp["s"])
        except ValueError:
            pass
    if "p" in qp and qp["p"]:
        st.session_state.participant_id = qp["p"]
    if "t" in qp and qp["t"]:
        decoded = _TOKEN_TO_MODE.get(qp["t"])
        if decoded:
            st.session_state.assigned_mode = decoded
    if "w" in qp:
        st.session_state.withdrawn = (qp["w"] == "1")
    if "confirm_wd" in qp:
        st.session_state.confirm_withdraw = (qp["confirm_wd"] == "1")

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.step == 0 and "s" in st.query_params:
    _restore_from_url()


def show_withdrawal_sidebar():
    if not st.session_state.confirm_withdraw:
        if st.query_params.get("confirm_wd", "0") == "1":
            st.session_state.confirm_withdraw = True

    if not st.session_state.confirm_withdraw:
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                "<p style='font-size:12px;color:#6b7280;margin:6px 0;padding-top:4px;'>"
                "You can leave this study at any time without penalty.</p>",
                unsafe_allow_html=True
            )
        with col_btn:
            if st.button("⚠ Withdraw", key="btn_withdraw_float",
                         help="Leave the study — your data will not be saved"):
                st.session_state.confirm_withdraw = True
                st.query_params["confirm_wd"] = "1"
                st.rerun()
        st.markdown(
            "<hr style='border-color:rgba(220,38,38,.2);margin:4px 0 12px;'>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:rgba(220,38,38,.08);border:1px solid "
            "rgba(220,38,38,.4);border-radius:10px;padding:12px 16px;margin-bottom:12px;'>"
            "<strong style='color:#f87171;'>Withdraw from the study?</strong> "
            "<span style='color:#9ca3af;font-size:13px;'>"
            "Your data will <strong style='color:#f87171;'>not</strong> be saved.</span>"
            "</div>",
            unsafe_allow_html=True
        )
        col_sp, col_yes, col_no, col_sp2 = st.columns([3, 1, 1, 3])
        with col_yes:
            if st.button("✅ Yes – withdraw", key="btn_confirm_wd"):
                st.session_state.withdrawn        = True
                st.session_state.confirm_withdraw = False
                st.query_params["withdrawn"]  = "1"
                st.query_params["confirm_wd"] = "0"
                _save_to_url()
                st.rerun()
        with col_no:
            if st.button("↩ Cancel", key="btn_cancel_wd"):
                st.session_state.confirm_withdraw = False
                st.query_params["confirm_wd"] = "0"
                st.rerun()
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


def show_progress(step: int):
    labels = ["Consent","Pre-test","Mood Check","Music Chat","Post-test","Complete"]
    st.markdown(
        f'<p class="progress-label">Step {step+1} of {len(labels)} — {labels[step]}</p>',
        unsafe_allow_html=True
    )
    bar = '<div class="progress-bar">'
    for i in range(len(labels)):
        cls = "done" if i < step else ("active" if i == step else "progress-step")
        bar += f'<div class="progress-step {cls}"></div>'
    bar += "</div>"
    st.markdown(bar, unsafe_allow_html=True)


def clean_artist(raw: str) -> str:
    cuts = re.compile(
        r'\s+(This\s|The\s+song|It\s+is|It\'s\s'
        r'|A\s+high|A\s+fun|A\s+classic|A\s+beautiful|A\s+catchy|A\s+upbeat'
        r'|A\s+powerful|A\s+soothing|A\s+popular|A\s+timeless|A\s+unique'
        r'|Which\s|That\s|Because\s|Since\s|While\s|When\s|whose\s'
        r'|is\s+a\s|is\s+an\s|has\s+a\s|features\s+a\s|known\s+for)',
        re.IGNORECASE
    )
    m = cuts.search(raw)
    if m:
        raw = raw[:m.start()]
    protected = re.sub(r'\b(ft|feat|jr|sr|mr|ms|dr|vs|etc)\.',
                       lambda x: x.group(0).replace('.', '\x00'),
                       raw, flags=re.IGNORECASE)
    parts = re.split(r'\.\s+[A-Z]', protected)
    raw = parts[0].replace('\x00', '.')
    raw = raw.strip().rstrip("*,–-").strip()
    words = raw.split()
    return " ".join(words[:7]).strip() if len(words) > 7 else raw


def extract_songs(text: str, max_songs: int = 10) -> list[dict]:
    songs, seen = [], set()
    lines = text.split('\n')

    for line in lines:
        if len(songs) >= max_songs:
            break
        line = line.strip()
        if not line:
            continue
        clean_line = re.sub(r'^\s*\d+[.)\]\s]\s*', '', line).strip()
        title, artist = None, None

        m = re.search(r'[‘]([^‘’]{2,70})[’]\s+by\s+(.+)',
                      clean_line, re.IGNORECASE)
        if m:
            title  = m.group(1).strip()
            artist = clean_artist(m.group(2))

        if not title:
            m = re.search(r'[“]([^“”]{2,70})[”]\s+by\s+(.+)',
                          clean_line, re.IGNORECASE)
            if m:
                title  = m.group(1).strip()
                artist = clean_artist(m.group(2))

        if not title:
            by_positions = [m.start() for m in re.finditer(r"\s+by\s+", clean_line, re.IGNORECASE)]
            for by_pos in by_positions:
                before = clean_line[:by_pos].strip()
                after  = clean_line[by_pos:].strip()
                after  = re.sub(r'^by\s+', '', after, flags=re.IGNORECASE)
                q_m = re.match(r"^'(.+)'\s*$", before)
                if q_m:
                    title  = q_m.group(1).strip()
                    artist = clean_artist(after)
                    break

        if not title:
            m = re.match(r'^([A-Z][^\n]{1,60}?)\s+by\s+(.+)$',
                         clean_line, re.IGNORECASE)
            if m:
                cand_title = m.group(1).strip().strip("'\"")
                if len(cand_title.split()) <= 8:
                    title  = cand_title
                    artist = clean_artist(m.group(2))

        if title and artist:
            title  = title.rstrip("*.,").strip()
            artist = artist.rstrip("*.,").strip()
            key    = title.lower()
            if (len(title) > 1 and len(title) < 80
                    and len(artist) > 1 and key not in seen):
                seen.add(key)
                songs.append({"title": title, "artist": artist})

    return songs[:max_songs]


def spotify_search(title: str, artist: str) -> dict | None:
    if sp is None:
        return None
    try:
        queries = [
            f"track:{title} artist:{artist}",
            f"{title} {artist}",
            f"{title}",
        ]
        for q in queries:
            res   = sp.search(q=q, type="track", limit=3)
            items = res["tracks"]["items"]
            if not items:
                continue
            artist_lower = artist.lower()
            for track in items:
                track_artists = [a["name"].lower() for a in track["artists"]]
                if any(artist_lower in a or a in artist_lower for a in track_artists):
                    return track
            return items[0]
    except Exception:
        pass
    return None


def youtube_url(title: str, artist: str) -> str:
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(title+' '+artist+' official audio')}"

def spotify_url(title: str, artist: str) -> str:
    return f"https://open.spotify.com/search/{urllib.parse.quote(title+' '+artist)}"


def render_song(song: dict, mood_profile: dict):
    title, artist = song["title"], song["artist"]
    st.markdown(f"**🎧 {title}**")
    st.markdown(
        f"<span style='color:#9ca3af;font-size:13px'>{artist}</span>",
        unsafe_allow_html=True
    )

    yt_url  = youtube_url(title, artist)
    sp_url  = spotify_url(title, artist)
    col_a, col_b = st.columns(2)

    track = spotify_search(title, artist)

    if track:
        real_name   = track["name"]
        real_artist = track["artists"][0]["name"]
        sp_url      = track["external_urls"]["spotify"]
        yt_url      = youtube_url(real_name, real_artist)

        imgs = track.get("album", {}).get("images", [])
        if imgs:
            st.image(imgs[0]["url"], width=70)

        with col_a:
            st.markdown(f"[▶ Spotify]({sp_url})")
        with col_b:
            st.markdown(f"[▶ YouTube]({yt_url})")

        if track.get("preview_url"):
            st.audio(track["preview_url"], format="audio/mp3")

        features = get_audio_features(track["id"])
        if features:
            render_audio_features(features, mood_profile)
    else:
        with col_a:
            st.markdown(f"[▶ Spotify ↗]({sp_url})")
        with col_b:
            st.markdown(f"[▶ YouTube ↗]({yt_url})")

    st.markdown("<hr style='border-color:rgba(139,92,246,.1);margin:8px 0;'>",
                unsafe_allow_html=True)


def call_llm(messages: list[dict]) -> str:
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1800,
        temperature=0.9,
    )
    return r.choices[0].message.content


_SOFT_DISTRESS = [
    "crying all day", "been crying", "can't stop crying", "nothing is going right",
    "everything is wrong", "i can't cope", "can't cope", "don't know what to do",
    "falling apart", "breaking down", "feel hopeless", "completely lost",
    "don't want to be here", "can't go on",
    "overwhelmed with everything", "nothing feels right", "really struggling",
    "can't handle", "losing my mind", "feel terrible", "feel awful",
    "everything is too much", "nothing is working", "at my limit",
    "rock bottom", "can't do this", "so tired of everything",
    "feel really bad", "feel so low", "can't keep going",
]

def is_soft_distress(text: str) -> bool:
    tl = text.lower()
    return any(phrase in tl for phrase in _SOFT_DISTRESS)

def is_crisis_response(text: str) -> bool:
    return "wellbeing service" in text.lower() or "speak to someone" in text.lower()


def save_results():
    r   = st.session_state.responses
    row = {
        "participant_id":   st.session_state.participant_id,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "assigned_mode":    st.session_state.assigned_mode,
        "num_turns":        st.session_state.num_turns,
        "age":              r.get("age",""),
        "gender":           r.get("gender",""),
        "ethnicity":        r.get("ethnicity",""),
        "music_use":        r.get("music_use",""),
        "ai_familiarity":   r.get("ai_familiarity",""),
        "pre_mood":         r.get("pre_mood",""),
        "pre_focus":        r.get("pre_focus",""),
        "pre_stress":       r.get("pre_stress",""),
        "ind_mood":         r.get("ind_mood",""),
        "ind_focus":        r.get("ind_focus",""),
        "ind_stress":       r.get("ind_stress",""),
        "induction_drop":   r.get("induction_mood_drop",""),
        "post_mood":        r.get("post_mood",""),
        "post_focus":       r.get("post_focus",""),
        "post_stress":      r.get("post_stress",""),
        "mood_change":      r.get("mood_change",""),
        "focus_change":     r.get("focus_change",""),
        "stress_change":    r.get("stress_change",""),
        "enjoyment":        r.get("enjoyment",""),
        "usability":        r.get("usability",""),
        "trust":            r.get("trust",""),
        "intention_to_use": r.get("intention_to_use",""),
        "tone_perceived":   r.get("tone_perceived",""),
        "persuasion_felt":  r.get("persuasion_felt",""),
        "noticed_diff":     r.get("noticed_diff",""),
        "comments":         str(r.get("comments","")).replace(",",";"),
    }
    df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, mode="a",
              header=not os.path.exists(CSV_PATH),
              index=False, quoting=1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 · CONSENT
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.withdrawn:
    st.title("You have withdrawn from the study")
    st.info(
        "**Your data has not been saved.**\n\n"
        "You have successfully withdrawn from this study. "
        "No information you entered has been recorded or stored anywhere.\n\n"
        "Thank you for your time. If you have any questions, "
        "please contact the researcher."
    )
    st.markdown(f"**Researcher contact:** {RESEARCHER_EMAIL}")
    if st.button("Start again as a new participant", type="primary"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.participant_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.query_params.clear()
        st.rerun()

elif st.session_state.step == 0:
    show_progress(0)

    st.markdown(
        f'<img src="{CONSENT_IMAGE}" alt="Calm music venue scene" '
        f'style="width:100%;height:280px;object-fit:cover;object-position:center 40%;'
        f'border-radius:16px;margin-bottom:1.5rem;display:block;">',
        unsafe_allow_html=True
    )

    st.title("🎵 MoodTune")
    st.markdown("#### Music Recommendation Study")
    st.markdown("---")

    with st.expander("📄 What is this study? (Please read before you start)", expanded=True):
        st.markdown(f"""
**What is this?**
You're about to use an AI music chatbot and tell us what you thought of it.
That's it. Nothing scary. Takes about 10 minutes.

**What happens, step by step:**
1. A few quick questions about you — 2 minutes
2. A short reading exercise to get you in the right headspace — 2 minutes
3. Chat with the music AI — tell it how you feel and it'll recommend songs — 4 minutes
4. A few quick questions about the experience — 2 minutes
5. A short debrief explaining what we were really looking at — 1 minute

**Your data is private.**
We don't collect your name or contact details. Your answers are saved as anonymous
numbers and text, only used for a university research project.
You can leave at any time — just click the Withdraw button that appears on every screen.
Your data is only saved when you reach the final Thank You screen.

**Is it safe?**
The reading exercise uses ordinary, everyday statements — nothing upsetting.
The music chatbot only recommends appropriate songs.
If you ever feel uncomfortable, just stop and reach out to someone you trust.

**Who to contact:** {RESEARCHER_EMAIL}
        """)

    st.markdown("**Please confirm all three statements:**")
    c1 = st.checkbox("I have read and understood the participant information sheet.")
    c2 = st.checkbox("I agree to take part and for my anonymous data to be used in this study.")
    c3 = st.checkbox("I understand I can withdraw at any time before the final screen.")

    if c1 and c2 and c3:
        if st.button("Begin study →", type="primary"):
            st.session_state.assigned_mode = random.choice(["Regular", "Persuasive"])
            st.session_state.step = 1
            _save_to_url()
            st.rerun()
    else:
        st.info("☝️ Please tick all three boxes to continue.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 · PRE-TEST
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    show_withdrawal_sidebar()
    show_progress(1)
    st.title("Before we begin")
    st.caption("A few quick questions — about 2 minutes.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age    = st.selectbox("Age group", ["18–20","21–23","24+"])
        gender = st.selectbox("Gender",    ["Male","Female","Non-binary","Prefer not to say"])
    with col2:
        ethnicity = st.selectbox("Ethnic background", ETHNICITY_OPTIONS)

    st.markdown("---")
    st.markdown("**How are you feeling right now?** (1 = very low · 10 = very high)")
    col_a, col_b, col_c = st.columns(3)
    with col_a: pre_mood   = st.slider("😊 Mood",   1, 10, 5)
    with col_b: pre_focus  = st.slider("🎯 Focus",  1, 10, 5)
    with col_c: pre_stress = st.slider("😟 Stress", 1, 10, 5)

    st.markdown("---")
    col_d, col_e = st.columns(2)
    with col_d:
        music_use = st.radio("How often do you use music to manage your mood?",
                              ["Never","Rarely","Sometimes","Often","Always"])
    with col_e:
        ai_familiarity = st.radio("How familiar are you with AI chatbots?",
                                   ["Not at all","Slightly","Moderately","Very","Extremely"])

    st.markdown("---")
    if st.button("Continue →", type="primary"):
        st.session_state.responses = {
            "age": age, "gender": gender, "ethnicity": ethnicity,
            "pre_mood": pre_mood, "pre_focus": pre_focus, "pre_stress": pre_stress,
            "music_use": music_use, "ai_familiarity": ai_familiarity,
        }
        st.session_state.step = 2
        _save_to_url()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 · MOOD INDUCTION (Velten, 1968; Martin, 1990)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    show_withdrawal_sidebar()
    show_progress(2)
    st.title("A short exercise before we start")
    st.markdown("---")
    st.info(
        "**A short reading exercise before the music.** "
        "Please read each statement below and take a moment to sit with it. "
        "There are no right or wrong answers — just read naturally and honestly."
    )
    st.caption(
        "This brief exercise helps create a consistent emotional starting point across all participants, "
        "making the study results more reliable. It uses mild, everyday statements — nothing clinical or distressing. "
        "The music assistant that follows is designed to help lift your mood. (Velten, 1968; Martin, 1990)"
    )
    st.markdown("---")

    VELTEN_STATEMENTS = [
        "Every now and then I feel a little tired and low on energy.",
        "Sometimes things feel slightly harder than they need to be.",
        "I've had moments recently where I felt a bit flat or uninspired.",
        "At times I find it difficult to feel excited about things.",
        "Occasionally I feel like I could do with a boost.",
    ]

    for i, (stmt, img_url, img_alt) in enumerate(
        zip(VELTEN_STATEMENTS, INDUCTION_IMAGES, INDUCTION_IMAGE_ALTS), 1
    ):
        st.markdown(
            f'''<div style="display:grid;grid-template-columns:1fr 1.6fr;gap:14px;
                margin:10px 0 14px;">
              <div>
                <img src="{img_url}" alt="{img_alt}"
                  style="width:100%;height:180px;object-fit:cover;
                         border-radius:12px;display:block;">
                <p style="font-size:10px;color:#6b7280;margin:5px 0 0;
                           text-align:center;">Photo: Unsplash (free licence)</p>
              </div>
              <div style="background:rgba(139,92,246,.09);
                          border-left:4px solid #8b5cf6;
                          border-radius:10px;padding:18px 22px;
                          display:flex;align-items:center;min-height:180px;">
                <p style="color:#e5e7eb;font-size:16px;line-height:1.75;
                           margin:0;font-style:italic;">
                  {i}.&nbsp; {stmt}
                </p>
              </div>
            </div>''',
            unsafe_allow_html=True
        )

    st.markdown("---")

    import time as _time
    _MIN_SECONDS = 30

    if "induction_start" not in st.session_state:
        st.session_state.induction_start = _time.time()
    if "induction_done" not in st.session_state:
        st.session_state.induction_done = False

    _elapsed   = int(_time.time() - st.session_state.induction_start)
    _remaining = max(0, _MIN_SECONDS - _elapsed)

    if _remaining > 0:
        st.markdown(
            f'<div style="background:rgba(139,92,246,.06);border-radius:10px;'
            f'padding:12px 18px;text-align:center;margin:12px 0;">'
            f'<p style="color:#6b7280;font-size:13px;margin:0;">'
            f'Take your time — read each statement and let it settle. 🎵</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        _time.sleep(1)
        st.rerun()
    else:
        _finished = st.checkbox(
            "✅  I've read all the statements above and I'm ready to continue.",
            key="induction_done_cb",
            value=st.session_state.induction_done,
        )
        if _finished:
            st.session_state.induction_done = True

        if st.session_state.induction_done:
            st.markdown("---")
            st.markdown("**After reading those statements, how are you feeling right now?**")
            st.caption("(1 = very low · 10 = very high)")
            col_a, col_b, col_c = st.columns(3)
            with col_a: ind_mood   = st.slider("😊 Mood",   1, 10, 5, key="ind_mood")
            with col_b: ind_focus  = st.slider("🎯 Focus",  1, 10, 5, key="ind_focus")
            with col_c: ind_stress = st.slider("😟 Stress", 1, 10, 5, key="ind_stress")
            st.markdown("")
            if st.button("Continue to music assistant →", type="primary"):
                st.session_state.responses["ind_mood"]   = ind_mood
                st.session_state.responses["ind_focus"]  = ind_focus
                st.session_state.responses["ind_stress"] = ind_stress
                st.session_state.responses["induction_mood_drop"] = (
                    st.session_state.responses.get("pre_mood", ind_mood) - ind_mood
                )
                for _k in ["induction_start", "induction_done"]:
                    st.session_state.pop(_k, None)
                st.session_state.step = 3
                _save_to_url()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 · CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    show_withdrawal_sidebar()
    show_progress(3)
    st.title("🎵 MoodTune")
    st.markdown("**Tell me how you're feeling and I'll find the right music for you.**")
    st.caption("Chat naturally with the assistant. When you are finished, a button will appear at the bottom.")
    st.markdown("---")

    current_mood_profile = MOOD_PROFILES.get(
        st.session_state.get("current_mood", "default"),
        MOOD_PROFILES["default"]
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("songs"):
                _profile = MOOD_PROFILES.get(
                    msg.get("mood_key", "default"), MOOD_PROFILES["default"]
                )
                _songs_h = msg["songs"]
                _collapse_h = len(_songs_h) > 4
                st.markdown(f"**Your recommendations ({len(_songs_h)} songs):**")
                for s in _songs_h:
                    if _collapse_h:
                        with st.expander(f"🎧 {s['title']} · {s['artist']}"):
                            render_song(s, _profile)
                    else:
                        render_song(s, _profile)

    if prompt := st.chat_input("How are you feeling right now?"):
        st.session_state.num_turns += 1
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        mood_profile = detect_mood(prompt)
        mood_key = next(
            (k for k, v in MOOD_PROFILES.items() if v is mood_profile), "default"
        )
        if mood_key == "default" and st.session_state.get("current_mood","default") != "default":
            mood_key     = st.session_state.current_mood
            mood_profile = MOOD_PROFILES[mood_key]
        st.session_state.current_mood = mood_key

        user_prefs = extract_user_preferences(prompt)

        followup_signals = [
            "you just", "those songs", "that song", "what were", "what are the names",
            "similar", "more like", "same genre", "tell me more", "what was",
            "the songs you", "you gave me", "you recommended", "those recommendations",
            "that list", "why", "how come", "only one", "just one", "only gave",
            "did not give", "didn't give", "didn't provide", "why did you", "explain",
            "what happened", "where are", "the links", "the others", "missing",
            "not all", "just two", "only two", "one link", "direct link",
        ]
        is_followup = (
            any(sig in prompt.lower() for sig in followup_signals)
            or len(st.session_state.messages) > 2
        )

        prev_songs = []
        for m in st.session_state.messages:
            if m["role"] == "assistant" and m.get("songs"):
                for s in m["songs"]:
                    title  = s.get("title", "")
                    artist = s.get("artist", "")
                    prev_songs.append(f'"{title}" by {artist}')

        mode   = st.session_state.assigned_mode
        system = build_system_prompt(mode, mood_profile, st.session_state.messages,
                                     user_prefs)

        if prev_songs:
            songs_list = "; ".join(prev_songs)
            system += (
                f"\n\nPREVIOUS RECOMMENDATIONS IN THIS CONVERSATION: {songs_list}. "
                "If the user asks about previous songs, refer to this list accurately. "
                "If the user asks for similar music, base your suggestions on these songs."
            )
        if is_followup:
            system += (
                "\n\nCRITICAL — THIS IS A FOLLOW-UP MESSAGE IN AN ONGOING CONVERSATION. "
                "The full conversation history is provided above. "
                "Read it carefully and respond DIRECTLY to what the user is asking about. "
                "Do NOT start fresh with a new mood question. "
                "Do NOT ignore the previous exchange. "
                "If the user is asking about previous recommendations, explain or list them. "
                "If the user asks why links were not provided, acknowledge this and "
                "list the previously recommended songs again WITH their Spotify and YouTube links. "
                "Always respond in context of the conversation — never pretend it is the first message."
            )

        history = [{"role": "system", "content": system}]
        for m in st.session_state.messages:
            if m["role"] in ("user", "assistant"):
                history.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            with st.spinner("Analysing your mood and finding music…"):
                try:
                    reply = call_llm(history)
                except Exception as e:
                    st.error(f"AI error: {e}")
                    st.stop()

                st.markdown(reply)
                songs = []
                _max  = min((user_prefs.get("requested_count") or 10) if user_prefs else 10, 10)
                if not is_crisis_response(reply):
                    songs = extract_songs(reply, max_songs=_max)
                    if songs:
                        st.markdown("---")
                        st.markdown(f"**Your recommendations ({len(songs)} songs):**")
                        _collapse = len(songs) > 4
                        if _collapse:
                            st.caption("💡 Tip: Audio analysis details are collapsed for readability. Click any song to expand.")
                        for s in songs:
                            if _collapse:
                                with st.expander(f"🎧 {s['title']} · {s['artist']}"):
                                    render_song(s, mood_profile)
                            else:
                                render_song(s, mood_profile)
                    else:
                        st.caption("🔍 Could not detect song names — see text above.")
                else:
                    st.warning(
                        "💛 If you're struggling, please reach out to your university "
                        "wellbeing service or a trusted person."
                    )

        st.session_state.messages.append({
            "role": "assistant", "content": reply,
            "songs": songs, "mood_key": mood_key,
            "max_songs": _max
        })

    if st.session_state.num_turns >= 1:
        all_songs_flat = []
        for m in st.session_state.messages:
            if m.get("songs"):
                all_songs_flat.extend(m["songs"])
        if all_songs_flat:
            with st.expander(f"💾 Save my playlist ({len(all_songs_flat)} songs) — open all on Spotify"):
                st.caption("Click each link to open the song in Spotify and add it to your own playlist.")
                for s in all_songs_flat:
                    sp_link = f"https://open.spotify.com/search/{urllib.parse.quote(s['title']+' '+s['artist'])}"
                    yt_link = youtube_url(s['title'], s['artist'])
                    st.markdown(
                        f"**{s['title']}** · *{s['artist']}* — "
                        f"[Spotify]({sp_link}) · [YouTube]({yt_link})"
                    )

    if st.session_state.num_turns >= 2:
        st.markdown(
            "<div style='margin-top:8px;padding-top:8px;"
            "border-top:1px solid rgba(139,92,246,.15);'></div>",
            unsafe_allow_html=True
        )
        col_space, col_done = st.columns([2, 1])
        with col_done:
            if st.button("✅ Done chatting – go to final questions",
                         type="primary", key="btn_done_chat"):
                st.session_state.step = 4
                _save_to_url()
                st.rerun()
        with col_space:
            st.caption("👆 Click when you have finished chatting.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 · POST-TEST
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    show_withdrawal_sidebar()
    show_progress(4)
    st.title("Final questions")
    st.caption("Almost done — about 2 minutes.")
    st.markdown("---")

    st.markdown("**How are you feeling RIGHT NOW?**")
    col_a, col_b, col_c = st.columns(3)
    with col_a: post_mood   = st.slider("😊 Mood",   1, 10, 5)
    with col_b: post_focus  = st.slider("🎯 Focus",  1, 10, 5)
    with col_c: post_stress = st.slider("😟 Stress", 1, 10, 5)

    st.markdown("---")
    st.markdown("**About your experience**")
    col1, col2 = st.columns(2)
    with col1:
        enjoyment = st.slider("Enjoyed the recommendations?", 1, 10, 5)
        usability = st.slider("Easy to use?",                 1, 10, 5)
    with col2:
        trust     = st.slider("Trusted the recommendations?", 1, 10, 5)
        intention = st.slider("Would use again?",             1, 10, 5)

    st.markdown("---")
    st.markdown("**About the assistant's style**")
    tone_perceived = st.radio(
        "How would you describe the way the assistant communicated with you?",
        ["Very cold / robotic","Neutral / informational",
         "Slightly warm / friendly","Very warm / encouraging","Unsure"],
    )
    persuasion_felt = st.slider(
        "To what extent did the assistant try to encourage or motivate you? "
        "(1 = not at all · 10 = very much)", 1, 10, 5
    )
    noticed_diff = st.radio(
        "Did you notice anything distinctive about how the assistant spoke?",
        ["Yes","No","Not sure"],
    )

    st.markdown("---")
    comments = st.text_area("Any other comments or feedback? (optional)")

    if st.button("Submit →", type="primary"):
        pre = st.session_state.responses.get
        st.session_state.responses.update({
            "post_mood":        post_mood,
            "post_focus":       post_focus,
            "post_stress":      post_stress,
            "mood_change":      post_mood   - pre("pre_mood",   post_mood),
            "focus_change":     post_focus  - pre("pre_focus",  post_focus),
            "stress_change":    post_stress - pre("pre_stress", post_stress),
            "enjoyment":        enjoyment,
            "usability":        usability,
            "trust":            trust,
            "intention_to_use": intention,
            "tone_perceived":   tone_perceived,
            "persuasion_felt":  persuasion_felt,
            "noticed_diff":     noticed_diff,
            "comments":         comments,
        })
        st.session_state.step = 5
        _save_to_url()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 · DEBRIEF + SAVE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    show_progress(5)
    st.title("✅ Thank you!")

    mode_label = st.session_state.assigned_mode
    badge      = "🟢" if mode_label == "Persuasive" else "🔵"

    st.subheader("Debrief – what this study was really about")
    st.info(
        f"**Thank you so much for taking part — it really means a lot.**\n\n"
        f"This study was actually investigating whether the **way an AI chatbot communicates** "
        f"affects how much music recommendations improve your mood. We used a cover story "
        f"(describing it as a music accuracy study) simply to make sure your responses "
        f"reflected a natural experience rather than what you thought we wanted to see. "
        f"This is completely standard in psychology research (Orne, 1962) and in no way "
        f"means you were misled about anything harmful.\n\n"
        f"You were randomly assigned to the **{badge} {mode_label} version**:\n\n"
        f"- **🔵 Regular** — clear, factual music recommendations.\n"
        f"- **🟢 Persuasive** — warm, encouraging recommendations using empathy and "
        f"personalisation (Cialdini, 1984; Fogg, 2009).\n\n"
        f"The short reading exercise at the start was a validated technique to help "
        f"create a consistent starting point across participants — its effects are "
        f"mild and temporary (Velten, 1968). The music that followed was designed to "
        f"lift your mood back up.\n\n"
        f"Your data is completely anonymous. If you have any concerns at all, "
        f"please reach out to the researcher — we genuinely want this to have been "
        f"a positive experience for you."
    )
    st.markdown(f"**Researcher contact:** {RESEARCHER_EMAIL}")

    try:
        save_results()
        st.success("✅ Your anonymous data has been saved.")
    except Exception as e:
        st.warning(f"Could not save to CSV: {e}")
        st.dataframe(pd.DataFrame([st.session_state.responses]))

    if os.path.exists(CSV_PATH):
        with st.expander("📊 Researcher view – all results"):
            try:
                full = pd.read_csv(CSV_PATH, quoting=1)
                st.dataframe(full)
                numeric_cols = [
                    "mood_change","focus_change","stress_change",
                    "enjoyment","trust","usability","intention_to_use","persuasion_felt"
                ]
                available = [c for c in numeric_cols if c in full.columns]
                if "assigned_mode" in full.columns and len(full) >= 2:
                    st.markdown("**Mean scores by condition**")
                    st.dataframe(
                        full.groupby("assigned_mode")[available]
                            .agg(["mean","count"]).round(2)
                    )
                else:
                    st.caption("Collect data from at least 2 participants to see comparisons.")
                st.download_button(
                    "⬇ Download CSV",
                    data=full.to_csv(index=False).encode("utf-8"),
                    file_name="moodtune_results.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    st.markdown("---")
    st.markdown("### 🎵 A little gift to end on a high note")
    st.caption(
        "Since the reading exercise earlier may have brought your mood down slightly, "
        "here are some uplifting songs to send you off feeling good. Think of it as a thank-you."
    )
    RESTORATION_SONGS = [
        ("Happy", "Pharrell Williams", "https://open.spotify.com/search/Happy%20Pharrell%20Williams"),
        ("Good as Hell", "Lizzo", "https://open.spotify.com/search/Good%20as%20Hell%20Lizzo"),
        ("Can't Stop the Feeling!", "Justin Timberlake", "https://open.spotify.com/search/Can't%20Stop%20the%20Feeling%20Justin%20Timberlake"),
        ("Best Day of My Life", "American Authors", "https://open.spotify.com/search/Best%20Day%20of%20My%20Life%20American%20Authors"),
        ("Walking on Sunshine", "Katrina and the Waves", "https://open.spotify.com/search/Walking%20on%20Sunshine"),
    ]
    rest_cols = st.columns(5)
    for i, (title, artist, link) in enumerate(RESTORATION_SONGS):
        with rest_cols[i]:
            yt = youtube_url(title, artist)
            st.markdown(
                f'<div style="background:rgba(139,92,246,.08);border-radius:10px;'
                f'padding:10px;text-align:center;font-size:12px;">'
                f'<strong>🎧 {title}</strong><br>'
                f'<span style="color:#9ca3af">{artist}</span><br><br>'
                f'<a href="{link}" target="_blank" style="color:#a78bfa">▶ Spotify</a> &nbsp;'
                f'<a href="{yt}" target="_blank" style="color:#a78bfa">▶ YouTube</a>'
                f'</div>',
                unsafe_allow_html=True
            )
    st.markdown("---")

    if st.button("New participant →", type="primary"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.participant_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.query_params.clear()
        st.rerun()