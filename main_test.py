# import streamlit as st
# import json, os
# import google.generativeai as genai

# # --- CONFIGURATION (hard‐coded for now) ---
# GENIE_KEY = "AIzaSyD4CpBQFyNEcmMnn67Efu0Ql27Pv5Y41e4"
# genai.configure(api_key=GENIE_KEY)

# # Load a Gemini model
# model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# # --- DATA FILE SETUP ---
# DATA_FILE = "data.json"
# if not os.path.exists(DATA_FILE):
#     initial = {
#         "users": [
#             {
#                 "email": "alice@example.com",
#                 "pw": "password",
#                 "name": "Alice",
#                 "xp": 0,
#                 "nextRankThreshold": 100,
#                 "badges": [],
#                 "rank": "Bronze"
#             }
#         ],
#         "quests": [
#             {
#                 "title": "Mock Interview Q1",
#                 "description": "Describe your biggest technical challenge.",
#                 "xpReward": 10
#             },
#             {
#                 "title": "Presentation Drill",
#                 "description": "Explain a complex topic in 2 minutes.",
#                 "xpReward": 15
#             }
#         ]
#     }
#     with open(DATA_FILE, "w") as f:
#         json.dump(initial, f, indent=2)

# with open(DATA_FILE, "r") as f:
#     DATA = json.load(f)

# USERS  = DATA["users"]
# QUESTS = DATA["quests"]

# def save_data():
#     with open(DATA_FILE, "w") as f:
#         json.dump(DATA, f, indent=2)

# # --- AI TUTOR ---
# def ask_ai(question: str, context: str = "") -> str:
#     prompt = f"You are an expert tutor—be concise and helpful.\n\nContext: {context}\nQuestion: {question}"
#     chat = model.start_chat()               # fresh chat with history=[]
#     response = chat.send_message(prompt)    # genai.ChatResponse
#     return response.text                    # the generated reply

# # --- STATE & AUTH ---
# if "user" not in st.session_state:
#     st.session_state.user = None

# def login(email, pw):
#     return next((u for u in USERS if u["email"]==email and u["pw"]==pw), None)

# def signup(email, pw, name):
#     if any(u["email"]==email for u in USERS): return None
#     new = {"email":email,"pw":pw,"name":name,"xp":0,
#            "nextRankThreshold":100,"badges":[],"rank":"Bronze"}
#     USERS.append(new); save_data(); return new

# # --- UI PAGES ---
# def show_login():
#     st.title("Log in")
#     e = st.text_input("Email"); p = st.text_input("Password", type="password")
#     if st.button("Log in"):
#         user = login(e, p)
#         if user: 
#             st.session_state.user = user; st.rerun()
#         else:
#             st.error("Invalid credentials")

# def show_signup():
#     st.title("Sign up")
#     e = st.text_input("Email", key="su_e")
#     p = st.text_input("Password", type="password", key="su_p")
#     n = st.text_input("Name", key="su_n")
#     if st.button("Create account"):
#         u = signup(e, p, n)
#         if u: st.success("Account created—please log in")
#         else: st.error("Email already taken")

# def show_dashboard():
#     u = st.session_state.user
#     st.header(f"Hello, {u['name']}!")
#     st.subheader(f"Rank: {u.get('rank','Bronze')}")
#     prog = u["xp"]/u["nextRankThreshold"]
#     st.progress(min(prog,1.0))
#     st.markdown("**Badges:** " + (", ".join(u.get("badges",[])) or "None"))

# def show_quests():
#     st.header("Quests")
#     for q in QUESTS:
#         if st.button(f"Start: {q['title']}"):
#             st.session_state.current_quest = q

#     if "current_quest" in st.session_state:
#         q = st.session_state.current_quest
#         st.subheader(q.get("title", "No Title"))
#         st.write(q.get("description", "No description available."))
#         #  st.subheader(q["title"]); st.write(q["description"])
#         ans = st.text_area("Your answer…", key="ans")
#         if st.button("Submit"):
#             u = st.session_state.user
#             u["xp"] += q["xpReward"]
#             st.success(f"+{q['xpReward']} XP (Total: {u['xp']})")

#             # rank‐up
#             thresholds = [100,300,600]
#             ranks      = ["Bronze","Silver","Gold","Platinum"]
#             for t,r in zip(thresholds,ranks[1:]):
#                 if u["xp"]>=t and u.get("rank","Bronze")!=r:
#                     u["rank"]=r; st.balloons(); st.success(f"🎉 Now {r} rank!")
#                     break

#             save_data()
#             del st.session_state["current_quest"]
#             st.rerun()

# def show_chat():
#     st.header("AI Tutor Chat")
#     msg = st.chat_input("Ask a question…")
#     if msg:
#         st.chat_message("user").write(msg)
#         ctx = st.session_state.get("current_quest",{}).get("description","")
#         resp = ask_ai(msg, context=ctx)
#         st.chat_message("assistant").write(resp)

# # --- MAIN FLOW ---
# st.sidebar.title("EduGame AI Tutor")
# if not st.session_state.user:
#     m = st.sidebar.radio("Mode",["Log in","Sign up"])
#     show_login() if m=="Log in" else show_signup()
# else:
#     p = st.sidebar.radio("Go to",["Dashboard","Quests","Chat"])
#     (show_dashboard if p=="Dashboard" else 
#      show_quests    if p=="Quests"    else 
#      show_chat)()

# import streamlit as st
# import json, os
# import random
# import google.generativeai as genai

# # --- CONFIGURATION ---
# GENIE_KEY = "AIzaSyD4CpBQFyNEcmMnn67Efu0Ql27Pv5Y41e4"
# genai.configure(api_key=GENIE_KEY)
# model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# # --- DATA FILES ---
# USER_DATA_FILE = "data.json"
# QUIZ_DATA_FILE = "peer_review_quiz_dataset.json"

# # --- Load or create user data ---
# if not os.path.exists(USER_DATA_FILE):
#     initial = {
#         "users": [
#             {
#                 "email": "alice@example.com",
#                 "pw": "password",
#                 "name": "Alice",
#                 "xp": 0,
#                 "nextRankThreshold": 100,
#                 "badges": [],
#                 "rank": "Bronze"
#             }
#         ]
#     }
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump(initial, f, indent=2)

# with open(USER_DATA_FILE, "r") as f:
#     DATA = json.load(f)

# USERS = DATA["users"]

# with open(QUIZ_DATA_FILE, "r") as f:
#     QUESTS = json.load(f)

# # --- Save user data ---
# def save_data():
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump(DATA, f, indent=2)

# # --- AI Tutor ---
# def ask_ai(question: str, context: str = "") -> str:
#     prompt = f"You are an expert tutor. Context: {context}\n\nQuestion: {question}"
#     chat = model.start_chat()
#     response = chat.send_message(prompt)
#     return response.text

# # --- Auth ---
# if "user" not in st.session_state:
#     st.session_state.user = None

# def login(email, pw):
#     return next((u for u in USERS if u["email"] == email and u["pw"] == pw), None)

# def signup(email, pw, name):
#     if any(u["email"] == email for u in USERS):
#         return None
#     new = {"email": email, "pw": pw, "name": name, "xp": 0,
#            "nextRankThreshold": 100, "badges": [], "rank": "Bronze"}
#     USERS.append(new)
#     save_data()
#     return new

# # --- UI Pages ---
# def show_login():
#     st.title("Log in")
#     e = st.text_input("Email")
#     p = st.text_input("Password", type="password")
#     if st.button("Log in"):
#         user = login(e, p)
#         if user:
#             st.session_state.user = user
#             st.rerun()
#         else:
#             st.error("Invalid credentials")

# def show_signup():
#     st.title("Sign up")
#     e = st.text_input("Email", key="su_e")
#     p = st.text_input("Password", type="password", key="su_p")
#     n = st.text_input("Name", key="su_n")
#     if st.button("Create account"):
#         u = signup(e, p, n)
#         if u:
#             st.success("Account created—please log in")
#         else:
#             st.error("Email already taken")

# def show_dashboard():
#     u = st.session_state.user
#     st.header(f"Hello, {u['name']}!")
#     st.subheader(f"Rank: {u.get('rank', 'Bronze')}")
#     prog = u["xp"] / u["nextRankThreshold"]
#     st.progress(min(prog, 1.0))
#     st.markdown("**Badges:** " + (", ".join(u.get("badges", [])) or "None"))

# def show_quests():
#     st.header("Quests")
#     for q in QUESTS:
#         if st.button(f"Start: {q['title']}"):
#             st.session_state.current_quest = q

#     if "current_quest" in st.session_state:
#         q = st.session_state.current_quest
#         st.subheader(q.get("title", "No Title"))
#         st.write(q.get("description", "No description available."))

#         st.markdown("#### Rubric")
#         for k, v in q.get("rubric", {}).items():
#             st.write(f"- **{k.title()}**: {v}")

#         st.markdown("#### AI Feedback Guide")
#         st.info(q.get("ai_feedback_guide", "No feedback guide provided."))

#         ans = st.text_area("Your answer…", key="ans")

#         if st.button("Submit"):
#             u = st.session_state.user
#             reward = q["xp_reward"]["submission"]
#             u["xp"] += reward
#             st.success(f"+{reward} XP (Total: {u['xp']})")

#             thresholds = [100, 300, 600, 1000]
#             ranks = ["Bronze", "Silver", "Gold", "Platinum", "Diamond"]
#             for t, r in zip(thresholds, ranks[1:]):
#                 if u["xp"] >= t and u.get("rank", "Bronze") != r:
#                     u["rank"] = r
#                     st.balloons()
#                     st.success(f"🎉 Rank Up! You’re now {r}.")
#                     break

#             save_data()
#             del st.session_state["current_quest"]
#             st.rerun()

# def show_chat():
#     st.header("AI Tutor Chat")
#     msg = st.chat_input("Ask a question…")
#     if msg:
#         st.chat_message("user").write(msg)
#         ctx = st.session_state.get("current_quest", {}).get("description", "")
#         resp = ask_ai(msg, context=ctx)
#         st.chat_message("assistant").write(resp)

# # --- MAIN FLOW ---
# st.sidebar.title("EduGame AI Tutor")
# if not st.session_state.user:
#     m = st.sidebar.radio("Mode", ["Log in", "Sign up"])
#     show_login() if m == "Log in" else show_signup()
# else:
#     p = st.sidebar.radio("Go to", ["Dashboard", "Quests", "Chat"])
#     (show_dashboard if p == "Dashboard" else
#      show_quests if p == "Quests" else
#      show_chat)()

# import streamlit as st
# import json, os, random, tempfile, datetime
# import google.generativeai as genai
# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
# import av
# import numpy as np
# import soundfile as sf

# # --- CONFIGURATION ---
# GENIE_KEY = "AIzaSyD4CpBQFyNEcmMnn67Efu0Ql27Pv5Y41e4"
# genai.configure(api_key=GENIE_KEY)
# model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# # --- DATA FILES ---
# USER_DATA_FILE = "data.json"
# QUIZ_DATA_FILE = "peer_review_quiz_dataset.json"
# HISTORY_LOG_FILE = "transcription_log.json"

# # --- Load or create user data ---
# if not os.path.exists(USER_DATA_FILE):
#     initial = {
#         "users": [
#             {
#                 "email": "alice@example.com",
#                 "pw": "password",
#                 "name": "Alice",
#                 "xp": 0,
#                 "nextRankThreshold": 100,
#                 "badges": [],
#                 "rank": "Bronze"
#             }
#         ]
#     }
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump(initial, f, indent=2)

# if not os.path.exists(HISTORY_LOG_FILE):
#     with open(HISTORY_LOG_FILE, "w") as f:
#         json.dump([], f, indent=2)

# with open(USER_DATA_FILE, "r") as f:
#     DATA = json.load(f)

# USERS = DATA["users"]

# if os.path.exists(QUIZ_DATA_FILE):
#     with open(QUIZ_DATA_FILE, "r") as f:
#         QUESTS = json.load(f)
# else:
#     QUESTS = []

# # --- Save user data ---
# def save_data():
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump(DATA, f, indent=2)

# # --- Save audio transcript log ---
# def log_transcription(entry):
#     with open(HISTORY_LOG_FILE, "r") as f:
#         log = json.load(f)
#     log.append(entry)
#     with open(HISTORY_LOG_FILE, "w") as f:
#         json.dump(log, f, indent=2)

# # --- Load user-specific transcripts ---
# def get_user_history(email):
#     with open(HISTORY_LOG_FILE, "r") as f:
#         log = json.load(f)
#     return [entry for entry in log if entry["user"] == email]

# # --- AI Tutor via audio ---
# def ask_ai_audio(audio_bytes, context=""):
#     prompt = "This is an audio-based question. Please transcribe it and respond informatively as an AI tutor."
#     if context:
#         prompt += f" The topic is: {context}"

#     response = model.generate_content([
#         {"mime_type": "audio/wav", "data": audio_bytes},
#         {"text": prompt}
#     ])
#     return response.text if hasattr(response, "text") else "Sorry, I couldn't understand the recording."

# # --- Streamlit Audio Recorder using WebRTC ---
# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.recorded_frames = []

#     def recv_queued(self, frames):
#         for frame in frames:
#             audio = frame.to_ndarray()
#             if audio.ndim > 1:
#                 audio = audio.mean(axis=0)
#             pcm = audio.astype(np.float32) / 32768.0
#             self.recorded_frames.append(pcm)
#         return frames[-1] if frames else None

# def get_audio_from_webrtc():
#     ctx = webrtc_streamer(
#         key="audio",
#         audio_processor_factory=AudioProcessor,
#         media_stream_constraints={"audio": True, "video": False},
#         async_processing=True
#     )

#     if ctx.audio_processor:
#         st.session_state.recorded_frames = ctx.audio_processor.recorded_frames

#     if st.button("Submit Recording"):
#         if "recorded_frames" in st.session_state:
#             pcm = np.concatenate(st.session_state.recorded_frames)
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                 sf.write(f.name, pcm, samplerate=48000, format='WAV', subtype='PCM_16')
#                 with open(f.name, "rb") as audio_file:
#                     return audio_file.read()
#         else:
#             st.warning("No audio captured yet.")
#     return None

# # --- Auth ---
# if "user" not in st.session_state:
#     st.session_state.user = None

# def login(email, pw):
#     return next((u for u in USERS if u["email"] == email and u["pw"] == pw), None)

# def signup(email, pw, name):
#     if any(u["email"] == email for u in USERS):
#         return None
#     new = {"email": email, "pw": pw, "name": name, "xp": 0,
#            "nextRankThreshold": 100, "badges": [], "rank": "Bronze"}
#     USERS.append(new)
#     save_data()
#     return new

# # --- UI Pages ---
# def show_login():
#     st.title("Log in")
#     e = st.text_input("Email")
#     p = st.text_input("Password", type="password")
#     if st.button("Log in"):
#         user = login(e, p)
#         if user:
#             st.session_state.user = user
#             st.rerun()
#         else:
#             st.error("Invalid credentials")

# def show_signup():
#     st.title("Sign up")
#     e = st.text_input("Email", key="su_e")
#     p = st.text_input("Password", type="password", key="su_p")
#     n = st.text_input("Name", key="su_n")
#     if st.button("Create account"):
#         u = signup(e, p, n)
#         if u:
#             st.success("Account created—please log in")
#         else:
#             st.error("Email already taken")

# def show_dashboard():
#     u = st.session_state.user
#     st.header(f"Hello, {u['name']}!")
#     st.subheader(f"Rank: {u.get('rank', 'Bronze')}")
#     prog = u["xp"] / u["nextRankThreshold"]
#     st.progress(min(prog, 1.0))
#     st.markdown("**Badges:** " + (", ".join(u.get("badges", [])) or "None"))

# def show_quests():
#     st.header("Quests")
#     for q in QUESTS:
#         if st.button(f"Start: {q.get('title', 'Untitled')}"):
#             st.session_state.current_quest = q

#     if "current_quest" in st.session_state:
#         q = st.session_state.current_quest
#         st.subheader(q.get("title", "No Title"))
#         st.write(q.get("description", "No description available."))

#         st.markdown("#### Rubric")
#         for k, v in q.get("rubric", {}).items():
#             st.write(f"- **{k.title()}**: {v}")

#         st.markdown("#### AI Feedback Guide")
#         st.info(q.get("ai_feedback_guide", "No feedback guide provided."))

#         st.write("### 🎙 Record your answer below")
#         audio_bytes = get_audio_from_webrtc()

#         if audio_bytes:
#             u = st.session_state.user
#             response_text = ask_ai_audio(audio_bytes, context=q.get("description", ""))
#             reward = q.get("xp_reward", {}).get("submission", 10)
#             u["xp"] += reward
#             st.success(f"+{reward} XP (Total: {u['xp']})")

#             thresholds = [100, 300, 600, 1000]
#             ranks = ["Bronze", "Silver", "Gold", "Platinum", "Diamond"]
#             for t, r in zip(thresholds, ranks[1:]):
#                 if u["xp"] >= t and u.get("rank", "Bronze") != r:
#                     u["rank"] = r
#                     st.balloons()
#                     st.success(f"🎉 Rank Up! You’re now {r}.")
#                     break

#             log_transcription({
#                 "user": u["email"],
#                 "context": q.get("title"),
#                 "timestamp": datetime.datetime.now().isoformat(),
#                 "response": response_text
#             })

#             st.markdown("#### AI Feedback")
#             st.info(response_text)

#             save_data()
#             del st.session_state["current_quest"]
#             st.rerun()

# def show_chat():
#     st.header("AI Tutor Chat")
#     st.write("🎙 Speak your question below")
#     audio_bytes = get_audio_from_webrtc()

#     if audio_bytes:
#         ctx = st.session_state.get("current_quest", {}).get("description", "")
#         resp = ask_ai_audio(audio_bytes, context=ctx)
#         st.markdown("**AI responded:**")
#         st.success(resp)

#         log_transcription({
#             "user": st.session_state.user["email"],
#             "context": "chat",
#             "timestamp": datetime.datetime.now().isoformat(),
#             "response": resp
#         })

# def show_history():
#     st.header("Chat History")
#     u = st.session_state.user
#     history = get_user_history(u["email"])

#     if not history:
#         st.info("No past interactions found.")
#         return

#     for item in reversed(history[-50:]):  # show latest 50 entries
#         st.markdown(f"**{item['timestamp']}**  ")
#         st.markdown(f"**Context:** {item['context']}")
#         st.markdown(f"**Response:** {item['response']}")
#         st.markdown("---")

# # --- MAIN FLOW ---
# st.sidebar.title("EduGame AI Tutor")
# if not st.session_state.user:
#     m = st.sidebar.radio("Mode", ["Log in", "Sign up"])
#     show_login() if m == "Log in" else show_signup()
# else:
#     p = st.sidebar.radio("Go to", ["Dashboard", "Quests", "Chat", "History"])
#     (show_dashboard if p == "Dashboard" else
#      show_quests if p == "Quests" else
#      show_chat if p == "Chat" else
#      show_history)()

# import streamlit as st
# import json
# import os, tempfile, datetime
# import io
# import numpy as np
# import sounddevice as sd
# import soundfile as sf
# import google.generativeai as genai
# import assemblyai as aai
# from gtts import gTTS

# # --- PAGE CONFIG & CUSTOM STYLING ---
# st.set_page_config(page_title="SpeakWise", page_icon="🎤", layout="centered")
# st.markdown(
#     """
#     <style>
#     /* Main container background */
#     .reportview-container, .main {
#         background-color: white;
#     }
#     /* Sidebar styling */
#     .sidebar .sidebar-content {
#         background-color: #4f826f;
#         color: white;
#     }
#     /* Header colors */
#     h1, h2, h3, .stHeader {
#         color: #4f826f;
#     }
#     /* Buttons */
#     button {
#         background-color: #7aac9a !important;
#         color: white !important;
#     }
#     /* Progress bar */
#     .stProgress > div > div {
#         background-color: #7aac9a;
#     }
#     /* Links */
#     a {
#         color: #4f826f;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --- CONFIGURATION ---
# GENIE_KEY = os.getenv("GENIE_KEY", "AIzaSyD4CpBQFyNEcmMnn67Efu0Ql27Pv5Y41e4")
# aai.settings.api_key = os.getenv("AIAAI_KEY", "7b1c719070dc47209812dbc3e6a0bfc4")
# genai.configure(api_key=GENIE_KEY)
# model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# # Set up AssemblyAI transcription with desired model
# config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1)
# transcriber = aai.Transcriber(config=config)

# # --- FILE PATHS ---
# USER_DATA_FILE = "data.json"
# HISTORY_LOG_FILE = "transcription_log.json"

# # --- INITIALIZE DATA FILES ---
# def ensure_data_file(path, initial_content):
#     if not os.path.exists(path):
#         with open(path, "w") as f:
#             json.dump(initial_content, f, indent=2)

# ensure_data_file(USER_DATA_FILE, {"users": []})
# ensure_data_file(HISTORY_LOG_FILE, [])

# # --- LOAD DATA ---
# with open(USER_DATA_FILE, "r") as f:
#     DATA = json.load(f)
# USERS = DATA.get("users", [])

# # --- SAVE DATA ---
# def save_user_data():
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump(DATA, f, indent=2)

# # --- LOGGING ---
# def log_transcription(entry):
#     with open(HISTORY_LOG_FILE, "r") as f:
#         log = json.load(f)
#     log.append(entry)
#     with open(HISTORY_LOG_FILE, "w") as f:
#         json.dump(log, f, indent=2)

# def get_user_history(email):
#     with open(HISTORY_LOG_FILE, "r") as f:
#         log = json.load(f)
#     return [e for e in log if e.get("user") == email]

# # --- AUDIO RECORDING ---
# def record_audio(duration=5, fs=16000):
#     st.info(f"Recording for {duration} seconds...")
#     try:
#         recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
#         sd.wait()
#     except Exception as e:
#         st.error(f"Recording failed: {e}")
#         return None
#     pcm16 = (recording.flatten() * 32767).astype('int16')
#     buf = io.BytesIO()
#     sf.write(buf, pcm16, samplerate=fs, format='WAV', subtype='PCM_16')
#     buf.seek(0)
#     return buf.read()

# # --- TRANSCRIPTION WITH AssemblyAI (dynamic clarity grade) ---
# def transcribe_audio(audio_bytes):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         f.write(audio_bytes)
#         temp_path = f.name

#     try:
#         transcript = transcriber.transcribe(temp_path)
#     except Exception as e:
#         st.error(f"Transcription error: {e}")
#         return None

#     if transcript.status == "error":
#         st.error(f"Transcription failed: {transcript.error}")
#         return None

#     # Collect confidences from words, if available
#     confidences = []
#     if hasattr(transcript, 'words'):
#         for w in transcript.words:
#             if hasattr(w, 'confidence'):
#                 confidences.append(w.confidence)
#     elif hasattr(transcript, 'segments'):
#         for seg in transcript.segments:
#             if hasattr(seg, 'confidence'):
#                 confidences.append(seg.confidence)

#     grade = int(sum(confidences) / len(confidences) * 100) if confidences else 0

#     return transcript.text, grade

# # --- TTS using gTTS ---
# def text_to_speech(text):
#     try:
#         tts = gTTS(text)
#         buf = io.BytesIO()
#         tts.write_to_fp(buf)
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         st.error(f"TTS error: {e}")
#         return None
    
# # --- AI RESPONSE ---
# def ai_answer(text, context=""):
#     prompt = f"You are SpeakWise — an AI interviewer coach. The user says: '{text}'."
#     if context:
#         prompt += f" The role is: {context}."
#     response = model.generate_content(prompt)
#     return response.text

# # --- AUTH & STATE INITIALIZATION ---
# if "user" not in st.session_state:
#     st.session_state.user = None
# for field, default in [
#     ("spoken_lang", "English"),
#     ("spoken_fluency", "B1"),
#     ("interview_lang", "English"),
#     ("interview_fluency", "B1")
# ]:
#     if field not in st.session_state:
#         st.session_state[field] = default

# # --- AUTH FUNCTIONS ---
# def login(email, pw):
#     user = next((u for u in USERS if u["email"] == email and u["pw"] == pw), None)
#     if user:
#         st.session_state.user = user
#         st.session_state.cv = user.get("cv")
#         st.session_state.job_desc = user.get("job_desc")
#         st.session_state.linkedin = user.get("linkedin")
#         for fld in ["spoken_lang","spoken_fluency","interview_lang","interview_fluency"]:
#             st.session_state[fld] = user.get(fld, st.session_state[fld])
#     return user

# def signup(email, pw, name):
#     if any(u["email"] == email for u in USERS):
#         return None
#     new = {
#         "email": email, "pw": pw, "name": name,
#         "xp": 0, "nextRankThreshold": 100, "badges": [], "rank": "Bronze",
#         "cv": None, "job_desc": "", "linkedin": "",
#         "spoken_lang": st.session_state.spoken_lang,
#         "spoken_fluency": st.session_state.spoken_fluency,
#         "interview_lang": st.session_state.interview_lang,
#         "interview_fluency": st.session_state.interview_fluency
#     }
#     USERS.append(new)
#     save_user_data()
#     return new


# def show_dashboard():
#     u = st.session_state.user
#     st.header(f"Welcome, {u['name']} 📈")
#     st.subheader(f"Rank: {u.get('rank','Bronze')}")
#     st.progress(min(u.get('xp',0)/u.get('nextRankThreshold',100),1.0))
#     st.markdown("**Badges:** " + (", ".join(u.get('badges',[])) or "None"))
#     st.markdown("---")

#     with st.form(key="dashboard_form"):
#         cv_file = st.file_uploader("Upload Your CV (PDF)", type=["pdf"], key="cv_uploader")
#         jd = st.text_area("Job Description", value=st.session_state.get("job_desc",""), key="jd_area")
#         ln = st.text_input("LinkedIn Profile URL", value=st.session_state.get("linkedin",""), key="ln_input")
#         st.markdown("---")
#         st.subheader("Languages & Fluency")
#         col1, col2 = st.columns(2)
#         with col1:
#             sl = st.selectbox("Spoken Language", ["English","Hebrew","Arabic"], key="sl_lang")
#             sf = st.selectbox("Spoken CEFR Level", ["A1","A2","B1","B2","C1","C2"], key="sl_cefr")
#         with col2:
#             il = st.selectbox("Interview Language", ["English","Hebrew","Arabic"], key="il_lang")
#             iflv = st.selectbox("Interview CEFR Level", ["A1","A2","B1","B2","C1","C2"], key="il_cefr")
#         submitted = st.form_submit_button("Submit")
#     if submitted:
#         if cv_file:
#             u["cv"] = cv_file.name
#             st.session_state.cv = cv_file.name
#             st.success("CV uploaded.")
#         u["job_desc"], st.session_state.job_desc = jd, jd
#         u["linkedin"], st.session_state.linkedin = ln, ln
#         u["spoken_lang"], st.session_state.spoken_lang = sl, sl
#         u["spoken_fluency"], st.session_state.spoken_fluency = sf, sf
#         u["interview_lang"], st.session_state.interview_lang = il, il
#         u["interview_fluency"], st.session_state.interview_fluency = iflv, iflv
#         save_user_data()
#         st.success("Profile updated!")

# def show_chat():
#     st.header("AI Interview Coach 🎤")
#     if st.button("🎤 Record 5s"):
#         audio = record_audio(5)
#         if audio:
#             st.audio(audio, format='audio/wav')
#             res = transcribe_audio(audio)
#             if not res: return
#             txt, grade = res
#             st.markdown("**Transcript:**")
#             st.write(txt)
#             st.markdown(f"**Clarity Grade:** {grade}/100")
#             reply = ai_answer(txt, context=st.session_state.get('job_desc',''))
#             st.markdown("**AI Feedback:**")
#             st.audio(text_to_speech(reply), format='audio/mp3')
#             log_transcription({
#                 'user': st.session_state.user['email'],
#                 'timestamp': datetime.datetime.now().isoformat(),
#                 'context': 'chat',
#                 'transcript': txt,
#                 'grade': grade,
#                 'response': reply
#             })

# def show_history():
#     st.header("Your Interaction History 📜")
#     hist = get_user_history(st.session_state.user['email'])
#     if not hist:
#         st.info("No past interactions.")
#         return
#     for entry in reversed(hist[-20:]):
#         st.markdown(f"**{entry['timestamp']}**")
#         st.markdown(f"- Transcript: {entry.get('transcript')}  ")
#         st.markdown(f"- Grade: {entry.get('grade')}  ")
#         st.markdown(f"- Feedback: {entry.get('response')}  ")
#         st.markdown("---")

# # --- MAIN FLOW ---
# st.sidebar.title("SpeakWise")
# if not st.session_state.user:
#     choice = st.sidebar.radio("Mode", ["Log in","Sign up"])
#     show_login() if choice == "Log in" else show_signup()
# else:
#     page = st.sidebar.radio("Go to", ["Dashboard","Chat","History"])
#     if page == "Dashboard": show_dashboard()
#     elif page == "Chat": show_chat()
#     else: show_history()


# import streamlit as st
# import json, os, tempfile, datetime, io
# import numpy as np
# import sounddevice as sd
# import soundfile as sf
# import google.generativeai as genai
# import assemblyai as aai
# from gtts import gTTS

# # --- PAGE CONFIG & CUSTOM STYLING ---
# st.set_page_config(page_title="SpeakWise", page_icon="🎤", layout="centered")
# st.markdown("""
# <style>
# .reportview-container, .main { background-color: white; }
# .sidebar .sidebar-content { background-color: #4f826f; color: white; }
# h1, h2, h3, .stHeader { color: #4f826f; }
# button { background-color: #7aac9a !important; color: white !important; }
# .stProgress > div > div { background-color: #7aac9a; }
# a { color: #4f826f; }
# </style>
# """, unsafe_allow_html=True)

# # --- CONFIGURATION ---
# GENIE_KEY = os.getenv("GENIE_KEY", "AIzaSyD4CpBQFyNEcmMnn67Efu0Ql27Pv5Y41e4")
# aai.settings.api_key = os.getenv("AIAAI_KEY", "7b1c719070dc47209812dbc3e6a0bfc4")
# genai.configure(api_key=GENIE_KEY)
# model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1))

# # --- FILES ---
# USER_DATA_FILE = "data.json"
# HISTORY_LOG_FILE = "transcription_log.json"

# def ensure_data_file(path, content): 
#     if not os.path.exists(path): 
#         with open(path, "w") as f: json.dump(content, f, indent=2)

# ensure_data_file(USER_DATA_FILE, {"users": []})
# ensure_data_file(HISTORY_LOG_FILE, [])

# with open(USER_DATA_FILE) as f: USERS = json.load(f).get("users", [])
# def save_user_data(): json.dump({"users": USERS}, open(USER_DATA_FILE, "w"), indent=2)

# def log_transcription(entry): 
#     with open(HISTORY_LOG_FILE) as f: log = json.load(f)
#     log.append(entry)
#     with open(HISTORY_LOG_FILE, "w") as f: json.dump(log, f, indent=2)

# def get_user_history(email): 
#     with open(HISTORY_LOG_FILE) as f: log = json.load(f)
#     return [e for e in log if e.get("user") == email]

# # --- AUDIO FUNCTIONS ---
# def record_audio(duration=5, fs=16000):
#     st.info(f"Recording for {duration} seconds...")
#     try:
#         rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
#         sd.wait()
#         pcm16 = (rec.flatten() * 32767).astype('int16')
#         buf = io.BytesIO()
#         sf.write(buf, pcm16, fs, format='WAV', subtype='PCM_16')
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         st.error(f"Recording failed: {e}")
#         return None

# def transcribe_audio(audio_buf):
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
#         f.write(audio_buf.read())
#         f.flush()
#         path = f.name
#     try:
#         transcript = transcriber.transcribe(path)
#         os.remove(path)
#     except Exception as e:
#         st.error(f"Transcription error: {e}")
#         return None
#     if transcript.status == 'error':
#         st.error(f"Transcription failed: {transcript.error}")
#         return None
#     confs = [w.confidence for w in getattr(transcript, 'words', []) if hasattr(w, 'confidence')]
#     grade = int(sum(confs) / len(confs) * 100) if confs else 0
#     return transcript.text, grade

# def text_to_speech(text):
#     try:
#         tts = gTTS(text)
#         buf = io.BytesIO()
#         tts.write_to_fp(buf)
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         st.error(f"TTS error: {e}")
#         return None

# def ai_answer(text, context=''):
#     prompt = f"You are SpeakWise — an AI interviewer coach. The user says: '{text}'."
#     if context:
#         prompt += f" The role is: {context}."
#     return model.generate_content(prompt).text

# # --- AUTH STATE ---
# if 'user' not in st.session_state: st.session_state.user = None
# for f, v in [('spoken_lang','English'),('spoken_fluency','B1'),('interview_lang','English'),('interview_fluency','B1')]:
#     if f not in st.session_state: st.session_state[f] = v

# def login(email, pw):
#     u = next((u for u in USERS if u['email']==email and u['pw']==pw), None)
#     if u:
#         st.session_state.user = u
#         for k in ['cv','job_desc','linkedin']: st.session_state[k] = u.get(k,'')
#         for k in ['spoken_lang','spoken_fluency','interview_lang','interview_fluency']:
#             st.session_state[k] = u.get(k, st.session_state[k])
#     return u

# def signup(email, pw, name):
#     if any(u['email']==email for u in USERS): return None
#     new = {'email':email,'pw':pw,'name':name,'xp':0,'nextRankThreshold':100,'badges':[],'rank':'Bronze',
#            'cv':'','job_desc':'','linkedin':'',
#            'spoken_lang':st.session_state.spoken_lang, 'spoken_fluency':st.session_state.spoken_fluency,
#            'interview_lang':st.session_state.interview_lang, 'interview_fluency':st.session_state.interview_fluency}
#     USERS.append(new)
#     save_user_data()
#     return new

# # --- PAGES ---
# def show_login():
#     st.title('Log in to SpeakWise')
#     e = st.text_input('Email')
#     p = st.text_input('Password', type='password')
#     if st.button('Log in') and login(e, p): st.experimental_rerun()
#     elif st.button('Log in'): st.error('Invalid credentials')

# def show_signup():
#     st.title('Sign up for SpeakWise')
#     e = st.text_input('Email', key='su_email')
#     p = st.text_input('Password', type='password', key='su_pw')
#     n = st.text_input('Name', key='su_name')
#     if st.button('Create account') and signup(e, p, n): st.success('Account created! Please log in.')

# def show_dashboard():
#     u = st.session_state.user
#     st.header(f"Welcome, {u['name']} 📈")
#     st.subheader(f"Rank: {u.get('rank','Bronze')}")
#     st.progress(min(u.get('xp',0)/u.get('nextRankThreshold',100),1.0))
#     st.markdown("**Badges:** " + ", ".join(u.get('badges',[])) or 'None')
#     with st.form('dash'):
#         cv = st.file_uploader('Upload Your CV (PDF)', type=['pdf'])
#         jd = st.text_area('Job Description', value=st.session_state.get('job_desc',''))
#         ln = st.text_input('LinkedIn URL', value=st.session_state.get('linkedin',''))
#         sl = st.selectbox('Spoken Language', ['English','Hebrew','Arabic'], index=['English','Hebrew','Arabic'].index(st.session_state.spoken_lang))
#         sf = st.selectbox('Spoken Level', ['A1','A2','B1','B2','C1','C2'], index=['A1','A2','B1','B2','C1','C2'].index(st.session_state.spoken_fluency))
#         il = st.selectbox('Interview Language', ['English','Hebrew','Arabic'], index=['English','Hebrew','Arabic'].index(st.session_state.interview_lang))
#         ifl = st.selectbox('Interview Level', ['A1','A2','B1','B2','C1','C2'], index=['A1','A2','B1','B2','C1','C2'].index(st.session_state.interview_fluency))
#         submit = st.form_submit_button('Submit')
#     if submit:
#         if cv:
#             st.session_state.cv = cv.name
#             u['cv'] = cv.name
#         u.update(job_desc=jd, linkedin=ln, spoken_lang=sl, spoken_fluency=sf, interview_lang=il, interview_fluency=ifl)
#         save_user_data()
#         st.success('Profile updated!')

# def show_chat():
#     st.header('AI Interview Coach 🎤')
#     if st.button('🎤 Record 5s'):
#         audio = record_audio()
#         if audio:
#             st.audio(audio, format='audio/wav')
#             transcript_result = transcribe_audio(audio)
#             if transcript_result:
#                 txt, grade = transcript_result
#                 st.markdown('**Transcript:**')
#                 st.write(txt)
#                 st.markdown(f'**Clarity Grade:** {grade}/100')
#                 reply = ai_answer(txt, context=st.session_state.get('job_desc',''))
#                 st.markdown('**AI Feedback:**')
#                 tts = text_to_speech(reply)
#                 if tts:
#                     st.audio(tts, format='audio/wav')
#                 log_transcription({
#                     'user': st.session_state.user['email'],
#                     'timestamp': datetime.datetime.now().isoformat(),
#                     'context': 'chat',
#                     'transcript': txt,
#                     'grade': grade,
#                     'response': reply
#                 })

# def show_history():
#     st.header('Your Interaction History 📜')
#     for e in reversed(get_user_history(st.session_state.user['email'])[-20:]):
#         st.markdown(f"**{e['timestamp']}**")
#         st.markdown(f"- Transcript: {e['transcript']}")
#         st.markdown(f"- Grade: {e['grade']}")
#         st.markdown(f"- Feedback: {e['response']}")
#         st.markdown("---")

# # --- MAIN ---
# st.sidebar.title('SpeakWise')
# if not st.session_state.user:
#     page = st.sidebar.radio('Mode', ['Log in', 'Sign up'])
#     show_login() if page == 'Log in' else show_signup()
# else:
#     page = st.sidebar.radio('Go to', ['Dashboard', 'Chat', 'History'])
#     {'Dashboard': show_dashboard, 'Chat': show_chat, 'History': show_history}[page]()

# import streamlit as st
# import json, os, tempfile, datetime, io
# import numpy as np
# import sounddevice as sd
# import soundfile as sf
# import google.generativeai as genai
# import assemblyai as aai
# from gtts import gTTS

# # Constants for XP rewards
# Y = 5  # Reward for scoring above average
# X = 10 # Daily leaderboard reward/penalty
# Z = 20 # Monthly leaderboard reward/penalty

# # --- PAGE CONFIG & CUSTOM STYLING ---
# st.set_page_config(page_title="SpeakWise", page_icon="🎤", layout="centered")
# st.markdown("""
# <style>
# .reportview-container, .main { background-color: white; }
# .sidebar .sidebar-content { background-color: #4f826f; color: white; }
# h1, h2, h3, .stHeader { color: #4f826f; }
# button { background-color: #7aac9a !important; color: white !important; }
# .stProgress > div > div { background-color: #7aac9a; }
# a { color: #4f826f; }
# </style>
# """, unsafe_allow_html=True)

# # --- CONFIGURATION ---
# GENIE_KEY = os.getenv("GENIE_KEY", "AIzaSyD4CpBQFyNEcmMnn67Efu0Ql27Pv5Y41e4")
# aai.settings.api_key = os.getenv("AIAAI_KEY", "7b1c719070dc47209812dbc3e6a0bfc4")
# genai.configure(api_key=GENIE_KEY)
# model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1))

# # --- FILE PATHS ---
# USER_DATA_FILE = "data.json"
# HISTORY_LOG_FILE = "transcription_log.json"
# COMMUNITY_FILE = "community.json"

# # --- INITIALIZE DATA FILES ---
# def ensure_data_file(path, content):
#     if not os.path.exists(path):
#         with open(path, "w") as f:
#             json.dump(content, f, indent=2)

# ensure_data_file(USER_DATA_FILE, {"users": []})
# ensure_data_file(HISTORY_LOG_FILE, [])
# ensure_data_file(COMMUNITY_FILE, [])

# # --- LOAD DATA ---
# with open(USER_DATA_FILE) as f:
#     USERS = json.load(f).get("users", [])
# # ensure xp fields exist for demo
# for u in USERS:
#     u.setdefault("xp_today", 0)
#     u.setdefault("xp_month", 0)

# # --- SAVE DATA ---
# def save_user_data():
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump({"users": USERS}, f, indent=2)

# # --- LOGGING ---
# def log_transcription(entry):
#     with open(HISTORY_LOG_FILE) as f:
#         log = json.load(f)
#     log.append(entry)
#     with open(HISTORY_LOG_FILE, "w") as f:
#         json.dump(log, f, indent=2)

# def get_user_history(email):
#     with open(HISTORY_LOG_FILE) as f:
#         log = json.load(f)
#     return [e for e in log if e.get("user") == email]

# # --- COMMUNITY ---
# def post_progress(user, message):
#     with open(COMMUNITY_FILE) as f:
#         posts = json.load(f)
#     posts.append({
#         "user": user["name"],
#         "timestamp": datetime.datetime.now().isoformat(),
#         "message": message
#     })
#     with open(COMMUNITY_FILE, "w") as f:
#         json.dump(posts, f, indent=2)

# def get_community_posts():
#     with open(COMMUNITY_FILE) as f:
#         return json.load(f)

# # --- AUDIO FUNCTIONS ---
# def record_audio(duration=5, fs=16000):
#     st.info(f"Recording for {duration} seconds...")
#     try:
#         rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
#         sd.wait()
#         pcm16 = (rec.flatten() * 32767).astype("int16")
#         buf = io.BytesIO()
#         sf.write(buf, pcm16, fs, format="WAV", subtype="PCM_16")
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         st.error(f"Recording failed: {e}")
#         return None

# def transcribe_audio(audio_buf):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         f.write(audio_buf.read())
#         f.flush()
#         path = f.name
#     try:
#         transcript = transcriber.transcribe(path)
#         os.remove(path)
#     except Exception as e:
#         st.error(f"Transcription error: {e}")
#         return None
#     if transcript.status == "error":
#         st.error(f"Transcription failed: {transcript.error}")
#         return None
#     # compute grade
#     confs = [w.confidence for w in getattr(transcript, "words", []) if hasattr(w, "confidence")]
#     grade = int(sum(confs) / len(confs) * 100) if confs else 0
#     return transcript.text, grade

# def text_to_speech(text):
#     try:
#         tts = gTTS(text)
#         buf = io.BytesIO()
#         tts.write_to_fp(buf)
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         st.error(f"TTS error: {e}")
#         return None

# def ai_answer(text, context=""):
#     prompt = f"You are SpeakWise — an AI interviewer coach. The user says: '{text}'."
#     if context:
#         prompt += f" The role is: {context}."
#     return model.generate_content(prompt).text

# # --- AUTH & STATE ---
# if "user" not in st.session_state:
#     st.session_state.user = None
# for fld, default in [
#     ("spoken_lang", "English"),
#     ("spoken_fluency", "B1"),
#     ("interview_lang", "English"),
#     ("interview_fluency", "B1")
# ]:
#     st.session_state.setdefault(fld, default)

# def login(email, pw):
#     user = next((u for u in USERS if u["email"] == email and u["pw"] == pw), None)
#     if user:
#         st.session_state.user = user
#         for key in ["cv", "job_desc", "linkedin", "spoken_lang", "spoken_fluency", "interview_lang", "interview_fluency"]:
#             st.session_state[key] = user.get(key, st.session_state.get(key))
#     return user

# def signup(email, pw, name):
#     if any(u["email"] == email for u in USERS):
#         return None
#     new = {
#         "email": email, "pw": pw, "name": name,
#         "xp": 0, "xp_today": 0, "xp_month": 0,
#         "nextRankThreshold": 100, "badges": [], "rank": "Bronze",
#         "cv": "", "job_desc": "", "linkedin": "",
#         "spoken_lang": st.session_state.spoken_lang,
#         "spoken_fluency": st.session_state.spoken_fluency,
#         "interview_lang": st.session_state.interview_lang,
#         "interview_fluency": st.session_state.interview_fluency
#     }
#     USERS.append(new)
#     save_user_data()
#     return new

# # --- PAGE FUNCTIONS ---
# def show_login():
#     st.title("Log in to SpeakWise")
#     email = st.text_input("Email")
#     pw = st.text_input("Password", type="password")
#     if st.button("Log in") and login(email, pw):
#         st.experimental_rerun()
#     elif st.button("Log in"):
#         st.error("Invalid credentials")

# def show_signup():
#     st.title("Sign up for SpeakWise")
#     email = st.text_input("Email", key="su_email")
#     pw = st.text_input("Password", type="password", key="su_pw")
#     name = st.text_input("Name", key="su_name")
#     if st.button("Create account") and signup(email, pw, name):
#         st.success("Account created! Please log in.")

# def show_dashboard():
#     u = st.session_state.user
#     st.header(f"Welcome, {u['name']} 📈")
#     st.subheader(f"Rank: {u.get('rank','Bronze')}")
#     st.progress(min(u.get("xp",0)/u.get("nextRankThreshold",100),1.0))
#     st.markdown("**Badges:** " + (", ".join(u.get("badges",[])) or "None"))
#     with st.form("dash"):
#         cv = st.file_uploader("Upload Your CV (PDF)", type=["pdf"])
#         jd = st.text_area("Job Description", value=st.session_state.get("job_desc",""))
#         ln = st.text_input("LinkedIn URL", value=st.session_state.get("linkedin",""))
#         sl = st.selectbox("Spoken Language", ["English","Hebrew","Arabic"], index=["English","Hebrew","Arabic"].index(st.session_state.spoken_lang))
#         sf = st.selectbox("Spoken Level", ["A1","A2","B1","B2","C1","C2"], index=["A1","A2","B1","B2","C1","C2"].index(st.session_state.spoken_fluency))
#         il = st.selectbox("Interview Language", ["English","Hebrew","Arabic"], index=["English","Hebrew","Arabic"].index(st.session_state.interview_lang))
#         ifl = st.selectbox("Interview Level", ["A1","A2","B1","B2","C1","C2"], index=["A1","A2","B1","B2","C1","C2"].index(st.session_state.interview_fluency))
#         submitted = st.form_submit_button("Submit")
#     if submitted:
#         if cv:
#             u["cv"] = cv.name
#             st.session_state.cv = cv.name
#         u.update(job_desc=jd, linkedin=ln, spoken_lang=sl, spoken_fluency=sf, interview_lang=il, interview_fluency=ifl)
#         save_user_data()
#         st.success("Profile updated!")


# # --- MAIN FLOW ---
# st.sidebar.title("SpeakWise")
# if not st.session_state.user:
#     mode = st.sidebar.radio("Mode", ["Log in", "Sign up"])
#     if mode == "Log in": show_login()
#     else: show_signup()
# else:
#     page = st.sidebar.radio("Go to", ["Dashboard","Chat","History","Community","Leaderboards"])
#     if page == "Dashboard": show_dashboard()
#     elif page == "Chat": show_chat()
#     elif page == "History": show_history()
#     elif page == "Community": show_community()
#     else: show_leaderboards()



# import streamlit as st
# import json, os, tempfile, datetime, io
# import numpy as np
# import sounddevice as sd
# import soundfile as sf
# import google.generativeai as genai
# import assemblyai as aai
# from gtts import gTTS

# # Constants for XP rewards
# Y = 5  # Reward for scoring above average
# X = 10 # Daily leaderboard reward/penalty
# Z = 20 # Monthly leaderboard reward/penalty

# # Badge icons mapping
# BADGE_ICONS = {
#     "Bronze": "🥉",
#     "Silver": "🥈",
#     "Gold": "🥇",
#     "Platinum": "🏆"
# }

# # --- PAGE CONFIG & CUSTOM STYLING ---
# st.set_page_config(page_title="SpeakWise", page_icon="🎤", layout="centered")
# st.markdown("""
# <style>
# .reportview-container, .main { background-color: white; }
# .sidebar .sidebar-content { background-color: #4f826f; color: white; }
# h1, h2, h3, .stHeader { color: #4f826f; }
# button { background-color: #7aac9a !important; color: white !important; }
# .stProgress > div > div { background-color: #7aac9a; }
# a { color: #4f826f; }
# </style>
# """, unsafe_allow_html=True)

# # --- CONFIGURATION ---
# GENIE_KEY = os.getenv("GENIE_KEY", "AIzaSyD4CpBQFyNEcmMnn67Efu0Ql27Pv5Y41e4")
# aai.settings.api_key = os.getenv("AIAAI_KEY", "7b1c719070dc47209812dbc3e6a0bfc4")
# genai.configure(api_key=GENIE_KEY)
# model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
# transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.slam_1))

# # --- FILE PATHS ---
# USER_DATA_FILE = "data.json"
# HISTORY_LOG_FILE = "transcription_log.json"
# COMMUNITY_FILE = "community.json"

# # --- INITIALIZE DATA FILES ---
# def ensure_data_file(path, content):
#     if not os.path.exists(path):
#         with open(path, "w") as f:
#             json.dump(content, f, indent=2)
# ensure_data_file(USER_DATA_FILE, {"users": []})
# ensure_data_file(HISTORY_LOG_FILE, [])
# ensure_data_file(COMMUNITY_FILE, [])

# # --- LOAD DATA ---
# with open(USER_DATA_FILE) as f:
#     USERS = json.load(f).get("users", [])
# for u in USERS:
#     u.setdefault("xp_today", 0)
#     u.setdefault("xp_month", 0)

# # --- SAVE DATA ---
# def save_user_data():
#     with open(USER_DATA_FILE, "w") as f:
#         json.dump({"users": USERS}, f, indent=2)

# # --- LOGGING ---
# def log_transcription(entry):
#     with open(HISTORY_LOG_FILE) as f:
#         log = json.load(f)
#     log.append(entry)
#     with open(HISTORY_LOG_FILE, "w") as f:
#         json.dump(log, f, indent=2)

# def get_user_history(email):
#     with open(HISTORY_LOG_FILE) as f:
#         log = json.load(f)
#     return [e for e in log if e.get("user") == email]

# # --- COMMUNITY ---
# def post_progress(user, message):
#     with open(COMMUNITY_FILE) as f:
#         posts = json.load(f)
#     posts.append({
#         "user": user["name"],
#         "timestamp": datetime.datetime.now().isoformat(),
#         "message": message
#     })
#     with open(COMMUNITY_FILE, "w") as f:
#         json.dump(posts, f, indent=2)

# def get_community_posts():
#     with open(COMMUNITY_FILE) as f:
#         return json.load(f)

# # --- AUDIO FUNCTIONS ---
# def record_audio(duration=5, fs=16000):
#     st.info(f"Recording for {duration} seconds...")
#     try:
#         rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
#         sd.wait()
#         pcm16 = (rec.flatten() * 32767).astype("int16")
#         buf = io.BytesIO()
#         sf.write(buf, pcm16, fs, format="WAV", subtype="PCM_16")
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         st.error(f"Recording failed: {e}")
#         return None

# def transcribe_audio(audio_buf):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         f.write(audio_buf.read())
#         f.flush()
#         path = f.name
#     try:
#         transcript = transcriber.transcribe(path)
#         os.remove(path)
#     except Exception as e:
#         st.error(f"Transcription error: {e}")
#         return None
#     if transcript.status == "error":
#         st.error(f"Transcription failed: {transcript.error}")
#         return None
#     confs = [w.confidence for w in getattr(transcript, "words", []) if hasattr(w, "confidence")]
#     grade = int(sum(confs) / len(confs) * 100) if confs else 0
#     return transcript.text, grade

# def text_to_speech(text):
#     try:
#         tts = gTTS(text)
#         buf = io.BytesIO()
#         tts.write_to_fp(buf)
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         st.error(f"TTS error: {e}")
#         return None

# def ai_answer(text, context=""):
#     prompt = f"You are SpeakWise — an AI interviewer coach. The user says: '{text}'."
#     if context:
#         prompt += f" The role is: {context}."
#     return model.generate_content(prompt).text

# # --- AUTH & STATE ---
# if "user" not in st.session_state:
#     st.session_state.user = None
# for fld, default in [("spoken_lang","English"),("spoken_fluency","B1"),("interview_lang","English"),("interview_fluency","B1")]:
#     st.session_state.setdefault(fld, default)

# def login(email, pw):
#     user = next((u for u in USERS if u["email"]==email and u["pw"]==pw), None)
#     if user:
#         st.session_state.user = user
#         for k in ["cv","job_desc","linkedin","spoken_lang","spoken_fluency","interview_lang","interview_fluency"]:
#             st.session_state[k] = user.get(k, st.session_state.get(k))
#     return user

# def signup(email, pw, name):
#     if any(u["email"]==email for u in USERS): return None
#     new = {"email":email,"pw":pw,"name":name,"xp":0,"xp_today":0,"xp_month":0,"nextRankThreshold":100,"badges":[],"rank":"Bronze","cv":"","job_desc":"","linkedin":"","spoken_lang":st.session_state.spoken_lang,"spoken_fluency":st.session_state.spoken_fluency,"interview_lang":st.session_state.interview_lang,"interview_fluency":st.session_state.interview_fluency}
#     USERS.append(new)
#     save_user_data()
#     return new

# # --- PAGE FUNCTIONS ---
# def show_login():
#     st.title("Log in to SpeakWise")
#     email = st.text_input("Email")
#     pw = st.text_input("Password", type="password")
    
#     login_clicked = st.button("Login", key="login_button")

#     if login_clicked:
#         if login(email, pw):
#             st.rerun()
#         else:
#             st.error("Invalid credentials")

# def show_signup():
#     st.title("Sign up for SpeakWise")
#     email = st.text_input("Email", key="su_email")
#     pw = st.text_input("Password", type="password", key="su_pw")
#     name = st.text_input("Name", key="su_name")
#     if st.button("Create account") and signup(email, pw, name): st.success("Account created! Please log in.")

# def show_dashboard():
#     u = st.session_state.user
#     st.header(f"Welcome, {u['name']} 📈")
#     st.subheader(f"Rank: {u.get('rank','Bronze')}")
#     st.progress(min(u.get("xp",0)/u.get("nextRankThreshold",100),1.0))
#     # Display badges with icons
#     badges = u.get("badges", [])
#     if badges:
#         display = [f"{BADGE_ICONS.get(b, '')} {b}" for b in badges]
#         st.markdown("**Badges:** " + ", ".join(display))
#     else:
#         st.markdown("**Badges:** None")
#     with st.form("dash"):
#         cv = st.file_uploader("Upload Your CV (PDF)", type=["pdf"])
#         jd = st.text_area("Job Description", value=st.session_state.get("job_desc",""))
#         ln = st.text_input("LinkedIn URL", value=st.session_state.get("linkedin",""))
#         sl = st.selectbox("Spoken Language", ["English","Hebrew","Arabic"], index=["English","Hebrew","Arabic"].index(st.session_state.spoken_lang))
#         sf = st.selectbox("Spoken Level", ["A1","A2","B1","B2","C1","C2"], index=["A1","A2","B1","B2","C1","C2"].index(st.session_state.spoken_fluency))
#         il = st.selectbox("Interview Language", ["English","Hebrew","Arabic"], index=["English","Hebrew","Arabic"].index(st.session_state.interview_lang))
#         ifl = st.selectbox("Interview Level", ["A1","A2","B1","B2","C1","C2"], index=["A1","A2","B1","B2","C1","C2"].index(st.session_state.interview_fluency))
#         submitted = st.form_submit_button("Submit")
#     if submitted:
#         if cv:
#             u["cv"] = cv.name
#             st.session_state.cv = cv.name
#         u.update(job_desc=jd, linkedin=ln, spoken_lang=sl, spoken_fluency=sf, interview_lang=il, interview_fluency=ifl)
#         save_user_data()
#         st.success("Profile updated!")

# def show_chat():
#     u = st.session_state.user
#     st.header("AI Interview Coach 🎤")
#     if st.button("🎤 Record 5s"):
#         audio_buf = record_audio()
#         if audio_buf:
#             st.audio(audio_buf, format="audio/wav")
#             result = transcribe_audio(audio_buf)
#             if result:
#                 txt, grade = result
#                 st.markdown("**Transcript:**")
#                 st.write(txt)
#                 st.markdown(f"**Clarity Grade:** {grade}/100")
#                 # reward if above average
#                 history = get_user_history(u["email"])
#                 grades = [e["grade"] for e in history if "grade" in e]
#                 avg_grade = sum(grades)/len(grades) if grades else grade
#                 if grade > avg_grade:
#                     st.success(f"Awarded {Y} XP for above-average performance!")
#                     u["xp"] += Y; u["xp_today"] += Y; u["xp_month"] += Y
#                 reply = ai_answer(txt, context=st.session_state.get("job_desc",""))
#                 st.markdown("**AI Feedback:**")
#                 tts_buf = text_to_speech(reply)
#                 if tts_buf:
#                     st.audio(tts_buf, format="audio/wav")
#                 log_transcription({
#                     "user": u["email"],
#                     "timestamp": datetime.datetime.now().isoformat(),
#                     "context": "chat",
#                     "transcript": txt,
#                     "grade": grade,
#                     "response": reply
#                 })
#                 save_user_data()

# def show_history():
#     st.header("Your Interaction History 📜")
#     for entry in reversed(get_user_history(st.session_state.user["email"]) [-20:]):
#         st.markdown(f"**{entry['timestamp']}**")
#         st.markdown(f"- Transcript: {entry.get('transcript')}")
#         st.markdown(f"- Grade: {entry.get('grade')}")
#         st.markdown(f"- Feedback: {entry.get('response')}")
#         st.markdown("---")

# def show_community():
#     st.header("Community Board 🌐")
#     message = st.text_area("Share your progress...")
#     if st.button("Post Update"):
#         if message.strip():
#             post_progress(st.session_state.user, message)
#             st.success("Posted!")
#         else:
#             st.error("Cannot post empty message.")
#     st.markdown("---")
#     st.subheader("Recent Updates")
#     for post in reversed(get_community_posts()[-20:]):
#         st.markdown(f"**{post['timestamp']}** - {post['user']}")
#         st.write(post['message'])
#         st.markdown("---")

# def show_leaderboards():
#     st.header("Leaderboards 🏆")
#     # Daily
#     st.subheader("Daily Leaderboard")
#     sorted_daily = sorted(USERS, key=lambda x: x.get("xp_today",0), reverse=True)
#     top3 = sorted_daily[:3]
#     bottom3 = sorted_daily[-3:] if len(sorted_daily)>=3 else sorted_daily[::-1]
#     st.write("Top 3 Users (Daily):")
#     for u in top3: st.write(f"{u['name']}: {u.get('xp_today',0)} XP")
#     st.write("Bottom 3 Users (Daily):")
#     for u in bottom3: st.write(f"{u['name']}: {u.get('xp_today',0)} XP")
#     if st.button("Apply Daily Rewards"):
#         for u in top3:
#             u["xp"] += X; u["xp_today"] += X; u["xp_month"] += X
#         for u in bottom3:
#             u["xp"] -= X; u["xp_today"] -= X; u["xp_month"] -= X
#         save_user_data()
#         st.success(f"Applied daily rewards/penalties: ±{X} XP")
#     # Monthly
#     st.subheader("Monthly Leaderboard")
#     sorted_monthly = sorted(USERS, key=lambda x: x.get("xp_month",0), reverse=True)
#     top3m = sorted_monthly[:3]
#     bottom3m = sorted_monthly[-3:] if len(sorted_monthly)>=3 else sorted_monthly[::-1]
#     st.write("Top 3 Users (Monthly):")
#     for u in top3m: st.write(f"{u['name']}: {u.get('xp_month',0)} XP")
#     st.write("Bottom 3 Users (Monthly):")
#     for u in bottom3m: st.write(f"{u['name']}: {u.get('xp_month',0)} XP")
#     if st.button("Apply Monthly Rewards"):
#         for u in top3m:
#             u["xp"] += Z; u["xp_today"] += Z; u["xp_month"] += Z
#         for u in bottom3m:
#             u["xp"] -= Z; u["xp_today"] -= Z; u["xp_month"] -= Z
#         save_user_data()
#         st.success(f"Applied monthly rewards/penalties: ±{Z} XP")


# # --- MAIN FLOW ---
# st.sidebar.title("SpeakWise")
# if not st.session_state.user:
#     mode = st.sidebar.radio("Mode", ["Log in", "Sign up"])
#     if mode == "Log in": show_login()
#     else: show_signup()
# else:
#     page = st.sidebar.radio("Go to", ["Dashboard","Chat","History","Community","Leaderboards"])
#     if page == "Dashboard": show_dashboard()
#     elif page == "Chat": show_chat()
#     elif page == "History": show_history()
#     elif page == "Community": show_community()
#     else: show_leaderboards()
