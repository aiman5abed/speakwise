
import streamlit as st
import json, os, tempfile, datetime, io
import numpy as np
import sounddevice as sd
import soundfile as sf
import google.generativeai as genai
import assemblyai as aai
from gtts import gTTS

# Constants for XP rewards
Y = 5  # Reward for scoring above average
X = 10 # Daily leaderboard reward/penalty
Z = 20 # Monthly leaderboard reward/penalty

# Badge icons mapping
BADGE_ICONS = {
    "Bronze": "🥉",
    "Silver": "🥈",
    "Gold": "🥇",
    "Platinum": "🏆"
}

# --- PAGE CONFIG & CUSTOM STYLING ---
st.set_page_config(page_title="SpeakWise", page_icon="🎤", layout="centered")
st.markdown("""
<style>
.reportview-container, .main { background-color: white; }
.sidebar .sidebar-content { background-color: #4f826f; color: white; }
h1, h2, h3, .stHeader { color: #4f826f; }
button { background-color: #7aac9a !important; color: white !important; }
.stProgress > div > div { background-color: #7aac9a; }
a { color: #4f826f; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
GENIE_KEY = os.getenv("GENIE_KEY", "AIzaSyD4CpBQFyNEcmMnn67Efu0Ql27Pv5Y41e4")
aai.settings.api_key = os.getenv("AIAAI_KEY", "7b1c719070dc47209812dbc3e6a0bfc4")
genai.configure(api_key=GENIE_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1))

# --- FILE PATHS ---
USER_DATA_FILE = "data.json"
HISTORY_LOG_FILE = "transcription_log.json"
COMMUNITY_FILE = "community.json"

# --- INITIALIZE DATA FILES ---
def ensure_data_file(path, content):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(content, f, indent=2)
ensure_data_file(USER_DATA_FILE, {"users": []})
ensure_data_file(HISTORY_LOG_FILE, [])
ensure_data_file(COMMUNITY_FILE, [])

# --- LOAD DATA ---
with open(USER_DATA_FILE) as f:
    USERS = json.load(f).get("users", [])
for u in USERS:
    u.setdefault("xp_today", 0)
    u.setdefault("xp_month", 0)

# --- SAVE DATA ---
def save_user_data():
    with open(USER_DATA_FILE, "w") as f:
        json.dump({"users": USERS}, f, indent=2)

# --- LOGGING ---
def log_transcription(entry):
    with open(HISTORY_LOG_FILE) as f:
        log = json.load(f)
    log.append(entry)
    with open(HISTORY_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def get_user_history(email):
    with open(HISTORY_LOG_FILE) as f:
        log = json.load(f)
    return [e for e in log if e.get("user") == email]

# --- COMMUNITY ---
def post_progress(user, message):
    with open(COMMUNITY_FILE) as f:
        posts = json.load(f)
    posts.append({
        "user": user["name"],
        "timestamp": datetime.datetime.now().isoformat(),
        "message": message
    })
    with open(COMMUNITY_FILE, "w") as f:
        json.dump(posts, f, indent=2)

def get_community_posts():
    with open(COMMUNITY_FILE) as f:
        return json.load(f)

# --- AUDIO FUNCTIONS ---
def record_audio(duration=10, fs=16000):
    st.info(f"Recording for {duration} seconds...")
    try:
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        pcm16 = (rec.flatten() * 32767).astype("int16")
        buf = io.BytesIO()
        sf.write(buf, pcm16, fs, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

def transcribe_audio(audio_buf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_buf.read())
        f.flush()
        path = f.name
    try:
        transcript = transcriber.transcribe(path)
        os.remove(path)
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None
    if transcript.status == "error":
        st.error(f"Transcription failed: {transcript.error}")
        return None
    confs = [w.confidence for w in getattr(transcript, "words", []) if hasattr(w, "confidence")]
    grade = int(sum(confs) / len(confs) * 100) if confs else 0
    return transcript.text, grade

def text_to_speech(text):
    try:
        tts = gTTS(text)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

def ai_answer(text, context=""):
    prompt = f"You are SpeakWise — an AI interviewer coach. The user says: '{text}'."
    if context:
        prompt += f" The role is: {context}."
    return model.generate_content(prompt).text

# --- AUTH & STATE ---
if "user" not in st.session_state:
    st.session_state.user = None
for fld, default in [("spoken_lang","English"),("spoken_fluency","B1"),("interview_lang","English"),("interview_fluency","B1")]:
    st.session_state.setdefault(fld, default)

def login(email, pw):
    user = next((u for u in USERS if u["email"]==email and u["pw"]==pw), None)
    if user:
        st.session_state.user = user
        for k in ["cv","job_desc","linkedin","spoken_lang","spoken_fluency","interview_lang","interview_fluency"]:
            st.session_state[k] = user.get(k, st.session_state.get(k))
    return user

def signup(email, pw, name):
    if any(u["email"]==email for u in USERS): return None
    new = {"email":email,"pw":pw,"name":name,"xp":0,"xp_today":0,"xp_month":0,"nextRankThreshold":100,"badges":[],"rank":"Bronze","cv":"","job_desc":"","linkedin":"","spoken_lang":st.session_state.spoken_lang,"spoken_fluency":st.session_state.spoken_fluency,"interview_lang":st.session_state.interview_lang,"interview_fluency":st.session_state.interview_fluency}
    USERS.append(new)
    save_user_data()
    return new

# --- PAGE FUNCTIONS ---
def show_login():
    st.title("Log in to SpeakWise")
    email = st.text_input("Email")
    pw = st.text_input("Password", type="password")
    
    login_clicked = st.button("Login", key="login_button")

    if login_clicked:
        if login(email, pw):
            st.rerun()
        else:
            st.error("Invalid credentials")

def show_signup():
    st.title("Sign up for SpeakWise")
    email = st.text_input("Email", key="su_email")
    pw = st.text_input("Password", type="password", key="su_pw")
    name = st.text_input("Name", key="su_name")
    if st.button("Create account") and signup(email, pw, name): st.success("Account created! Please log in.")

def show_dashboard():
    u = st.session_state.user
    st.header(f"Welcome, {u['name']} 📈")
    st.subheader(f"Rank: {u.get('rank','Bronze')}")
    st.progress(min(u.get("xp",0)/u.get("nextRankThreshold",100),1.0))
    # Display badges with icons
    badges = u.get("badges", [])
    if badges:
        display = [f"{BADGE_ICONS.get(b, '')} {b}" for b in badges]
        st.markdown("**Badges:** " + ", ".join(display))
    else:
        st.markdown("**Badges:** None")
    with st.form("dash"):
        cv = st.file_uploader("Upload Your CV (PDF)", type=["pdf"])
        jd = st.text_area("Job Description", value=st.session_state.get("job_desc",""))
        ln = st.text_input("LinkedIn URL", value=st.session_state.get("linkedin",""))
        sl = st.selectbox("Spoken Language", ["English","Hebrew","Arabic"], index=["English","Hebrew","Arabic"].index(st.session_state.spoken_lang))
        sf = st.selectbox("Spoken Level", ["A1","A2","B1","B2","C1","C2"], index=["A1","A2","B1","B2","C1","C2"].index(st.session_state.spoken_fluency))
        il = st.selectbox("Interview Language", ["English","Hebrew","Arabic"], index=["English","Hebrew","Arabic"].index(st.session_state.interview_lang))
        ifl = st.selectbox("Interview Level", ["A1","A2","B1","B2","C1","C2"], index=["A1","A2","B1","B2","C1","C2"].index(st.session_state.interview_fluency))
        submitted = st.form_submit_button("Submit")
    if submitted:
        if cv:
            u["cv"] = cv.name
            st.session_state.cv = cv.name
        u.update(job_desc=jd, linkedin=ln, spoken_lang=sl, spoken_fluency=sf, interview_lang=il, interview_fluency=ifl)
        save_user_data()
        st.success("Profile updated!")

def show_chat():
    u = st.session_state.user
    st.header("AI Interview Coach 🎤")
    if st.button("🎤 Record 5s"):
        audio_buf = record_audio()
        if audio_buf:
            st.audio(audio_buf, format="audio/wav")
            result = transcribe_audio(audio_buf)
            if result:
                txt, grade = result
                st.markdown("**Transcript:**")
                st.write(txt)
                st.markdown(f"**Clarity Grade:** {grade}/100")
                # reward if above average
                history = get_user_history(u["email"])
                grades = [e["grade"] for e in history if "grade" in e]
                avg_grade = sum(grades)/len(grades) if grades else grade
                if grade > avg_grade:
                    st.success(f"Awarded {Y} XP for above-average performance!")
                    u["xp"] += Y; u["xp_today"] += Y; u["xp_month"] += Y
                reply = ai_answer(txt, context=st.session_state.get("job_desc",""))
                st.markdown("**AI Feedback:**")
                tts_buf = text_to_speech(reply)
                if tts_buf:
                    st.audio(tts_buf, format="audio/wav")
                log_transcription({
                    "user": u["email"],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "context": "chat",
                    "transcript": txt,
                    "grade": grade,
                    "response": reply
                })
                save_user_data()

def show_history():
    st.header("Your Interaction History 📜")
    for entry in reversed(get_user_history(st.session_state.user["email"]) [-20:]):
        st.markdown(f"**{entry['timestamp']}**")
        st.markdown(f"- Transcript: {entry.get('transcript')}")
        st.markdown(f"- Grade: {entry.get('grade')}")
        st.markdown(f"- Feedback: {entry.get('response')}")
        st.markdown("---")

def show_community():
    st.header("Community Board 🌐")
    message = st.text_area("Share your progress...")
    if st.button("Post Update"):
        if message.strip():
            post_progress(st.session_state.user, message)
            st.success("Posted!")
        else:
            st.error("Cannot post empty message.")
    st.markdown("---")
    st.subheader("Recent Updates")
    for post in reversed(get_community_posts()[-20:]):
        st.markdown(f"**{post['timestamp']}** - {post['user']}")
        st.write(post['message'])
        st.markdown("---")

def show_leaderboards():
    st.header("Leaderboards 🏆")
    # Daily
    st.subheader("Daily Leaderboard")
    sorted_daily = sorted(USERS, key=lambda x: x.get("xp_today",0), reverse=True)
    top3 = sorted_daily[:3]
    bottom3 = sorted_daily[-3:] if len(sorted_daily)>=3 else sorted_daily[::-1]
    st.write("Top 3 Users (Daily):")
    for u in top3: st.write(f"{u['name']}: {u.get('xp_today',0)} XP")
    st.write("Bottom 3 Users (Daily):")
    for u in bottom3: st.write(f"{u['name']}: {u.get('xp_today',0)} XP")
    if st.button("Apply Daily Rewards"):
        for u in top3:
            u["xp"] += X; u["xp_today"] += X; u["xp_month"] += X
        for u in bottom3:
            u["xp"] -= X; u["xp_today"] -= X; u["xp_month"] -= X
        save_user_data()
        st.success(f"Applied daily rewards/penalties: ±{X} XP")
    # Monthly
    st.subheader("Monthly Leaderboard")
    sorted_monthly = sorted(USERS, key=lambda x: x.get("xp_month",0), reverse=True)
    top3m = sorted_monthly[:3]
    bottom3m = sorted_monthly[-3:] if len(sorted_monthly)>=3 else sorted_monthly[::-1]
    st.write("Top 3 Users (Monthly):")
    for u in top3m: st.write(f"{u['name']}: {u.get('xp_month',0)} XP")
    st.write("Bottom 3 Users (Monthly):")
    for u in bottom3m: st.write(f"{u['name']}: {u.get('xp_month',0)} XP")
    if st.button("Apply Monthly Rewards"):
        for u in top3m:
            u["xp"] += Z; u["xp_today"] += Z; u["xp_month"] += Z
        for u in bottom3m:
            u["xp"] -= Z; u["xp_today"] -= Z; u["xp_month"] -= Z
        save_user_data()
        st.success(f"Applied monthly rewards/penalties: ±{Z} XP")


# --- MAIN FLOW ---
st.sidebar.title("SpeakWise")
if not st.session_state.user:
    mode = st.sidebar.radio("Mode", ["Log in", "Sign up"])
    if mode == "Log in": show_login()
    else: show_signup()
else:
    page = st.sidebar.radio("Go to", ["Dashboard","Chat","History","Community","Leaderboards"])
    if page == "Dashboard": show_dashboard()
    elif page == "Chat": show_chat()
    elif page == "History": show_history()
    elif page == "Community": show_community()
    else: show_leaderboards()

