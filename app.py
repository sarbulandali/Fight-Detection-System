import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import tempfile
import os
from io import BytesIO
import wave

st.set_page_config(page_title="FightSense AI", page_icon="🛡️", layout="wide")

#  CUSTOM CSS — DARK THEME

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2230;
    --accent:    #e8ff47;
    --danger:    #ff3b3b;
    --safe:      #00e5a0;
    --warn:      #ffaa00;
    --text:      #e8eaf0;
    --muted:     #5a5f72;
    --card-bg:   #13161f;
}

html, body, [class*="css"] { font-family: 'Space Mono', monospace; background-color: var(--bg); color: var(--text); }
.stApp { background-color: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }

.app-header { display: flex; align-items: center; gap: 16px; padding: 24px 0 8px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
.app-header .logo { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; color: var(--accent); line-height: 1; }
.app-header .sub { font-size: 0.72rem; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; }

.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
.badge-danger { background: rgba(255,59,59,0.15); color: var(--danger); border: 1px solid var(--danger); }
.badge-safe  { background: rgba(0,229,160,0.12); color: var(--safe); border: 1px solid var(--safe); }
.badge-warn  { background: rgba(255,170,0,0.12); color: var(--warn); border: 1px solid var(--warn); }

.metric-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 18px 20px; text-align: center; }
.metric-card .val { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; line-height: 1; }
.metric-card .label { font-size: 0.65rem; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-top: 6px; }
.metric-card.danger .val { color: var(--danger); }
.metric-card.safe .val { color: var(--safe); }
.metric-card.warn .val { color: var(--warn); }
.metric-card.accent .val { color: var(--accent); }

.alert-fight { background: rgba(255,59,59,0.08); border: 1px solid var(--danger); border-left: 4px solid var(--danger); border-radius: 6px; padding: 16px 20px; margin: 12px 0; animation: pulse-border 1.5s ease-in-out infinite; }
.alert-safe { background: rgba(0,229,160,0.06); border: 1px solid var(--safe); border-left: 4px solid var(--safe); border-radius: 6px; padding: 16px 20px; margin: 12px 0; }
@keyframes pulse-border { 0%,100% { border-color: var(--danger); } 50% { border-color: #ff6b6b; box-shadow: 0 0 12px 2px rgba(255,59,59,0.25); } }

.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 8px; padding: 4px; gap: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important; letter-spacing: 1px !important; text-transform: uppercase !important; color: var(--muted) !important; background: transparent !important; border-radius: 6px !important; padding: 8px 20px !important; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: #0a0c10 !important; }

.stButton > button { font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important; font-weight: 700 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; border-radius: 6px !important; border: 1px solid var(--accent) !important; background: transparent !important; color: var(--accent) !important; padding: 10px 24px !important; transition: all 0.2s ease !important; }
.stButton > button:hover { background: var(--accent) !important; color: #0a0c10 !important; }
.stButton > button[kind="primary"] { background: var(--accent) !important; color: #0a0c10 !important; }

[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }
.stSelectbox label, .stSlider label { font-size: 0.72rem !important; letter-spacing: 1px; text-transform: uppercase; color: var(--muted) !important; }
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

[data-testid="stFileUploader"] { border: 1px dashed var(--border) !important; border-radius: 8px !important; background: var(--card-bg) !important; padding: 8px !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

.section-label { font-size: 0.65rem; letter-spacing: 3px; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.event-row { display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border); margin: 4px 0; background: var(--card-bg); font-size: 0.78rem; }
.event-row.fight { border-left: 3px solid var(--danger); }
.event-row.safe  { border-left: 3px solid var(--safe); }
.event-ts { color: var(--muted); font-size: 0.68rem; min-width: 54px; }
.conf-bar-wrap { margin: 10px 0; }
.conf-bar-wrap .bar-label { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); margin-bottom: 4px; letter-spacing: 1px; }
.conf-bar-bg { background: var(--border); border-radius: 3px; height: 6px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


#  MODEL LOADING (AUTO — NO BUTTON NEEDED)

SEQUENCE_LENGTH = 16
IMG_SIZE = 224

@st.cache_resource
def load_models():
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet', include_top=False, pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    lstm = tf.keras.models.load_model('best_fight_model.h5')
    return base_model, lstm

@st.cache_data
def get_alarm():
    if os.path.exists('alarm.mp3'):
        with open('alarm.mp3', 'rb') as f:
            return f.read(), "audio/mp3"
    # fallback beep
    sr, dur, freq = 44100, 0.5, 800
    t = np.linspace(0, dur, int(sr*dur), False)
    wav = (32767*np.sin(2*np.pi*freq*t)).astype(np.int16)
    buf = BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(wav.tobytes())
    return buf.getvalue(), "audio/wav"

# Auto‑load on startup
with st.spinner("Loading models..."):
    base_model, lstm_model = load_models()
alarm_bytes, alarm_fmt = get_alarm()


#  INFERENCE FUNCTIONS

def extract_features(rgb):
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    pre = tf.keras.applications.mobilenet_v2.preprocess_input(resized.astype(np.float32))
    return base_model.predict(np.expand_dims(pre,0), verbose=0).squeeze()

def predict(frame_bgr, buf):
    feat = extract_features(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    buf.append(feat)
    if len(buf) < SEQUENCE_LENGTH:
        return {"fight_prob":0.0, "safe_prob":1.0, "label":"SAFE", "confidence":0.0}
    seq = np.expand_dims(np.array(list(buf)), 0)
    fp = float(lstm_model.predict(seq, verbose=0)[0][0])
    sp = 1-fp
    return {"fight_prob":round(fp,3), "safe_prob":round(sp,3),
            "label":"FIGHT" if fp>0.5 else "SAFE", "confidence":round(max(fp,sp),3)}

def overlay(frame, pred, fps):
    h,w = frame.shape[:2]
    out = frame.copy()
    ov = out.copy()
    cv2.rectangle(ov,(0,0),(w,52),(10,12,16),-1)
    cv2.addWeighted(ov,0.7,out,0.3,0,out)
    clr = (59,59,255) if pred["label"]=="FIGHT" else (0,229,160)
    cv2.putText(out,pred["label"],(12,34),cv2.FONT_HERSHEY_DUPLEX,1.0,clr,2)
    bx,by,bw = 160,20,180
    cv2.rectangle(out,(bx,by),(bx+bw,by+12),(30,35,50),-1)
    cv2.rectangle(out,(bx,by),(bx+int(bw*pred["confidence"]),by+12),clr,-1)
    cv2.putText(out,f"{int(pred['confidence']*100)}%",(bx+bw+8,by+11),cv2.FONT_HERSHEY_PLAIN,0.9,clr,1)
    cv2.putText(out,f"FPS {fps:.1f}",(w-90,28),cv2.FONT_HERSHEY_PLAIN,1.0,(90,95,114),1)
    if pred["label"]=="FIGHT":
        cv2.rectangle(out,(4,4),(w-4,h-4),(59,59,255),4)
    return out


#  SESSION STATE

for k,v in {"running":False,"fight_count":0,"frame_count":0,"events":[],
            "last_pred":None,"buf":deque(maxlen=SEQUENCE_LENGTH),
            "fps_hist":deque(maxlen=30),"alarm_playing":False,"alarm_ctr":0}.items():
    if k not in st.session_state: st.session_state[k]=v


#  SIDEBAR

with st.sidebar:
    st.markdown("""<div style="padding:12px 0 20px;">
    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#e8ff47;">⚡ FightSense</div>
    <div style="font-size:0.6rem;color:#5a5f72;letter-spacing:3px;">AI Surveillance</div></div>""",unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model</div>',unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#00e5a0;">● MobileNetV2 + LSTM loaded</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">Settings</div>',unsafe_allow_html=True)
    thresh = st.slider("Confidence threshold",0.5,0.99,0.65,0.01)
    skip = st.slider("Frame skip",1,10,3)
    alarm_on = st.checkbox("🔊 Alarm",True)
    st.markdown("---")
    total = st.session_state.frame_count
    fights = st.session_state.fight_count
    rate = (fights/total*100) if total else 0
    st.markdown('<div class="section-label">Session</div>',unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;gap:8px;">
    <div class="metric-card danger"><div class="val">{fights}</div><div class="label">Fight Frames</div></div>
    <div class="metric-card accent"><div class="val">{total}</div><div class="label">Total Frames</div></div>
    <div class="metric-card {'danger' if rate>30 else 'safe'}"><div class="val">{rate:.1f}%</div><div class="label">Fight Rate</div></div>
    </div>""",unsafe_allow_html=True)
    if st.button("🔄 Reset",use_container_width=True):
        for k in ["fight_count","frame_count","events","last_pred","alarm_playing"]:
            st.session_state[k]=0 if k in ["fight_count","frame_count"] else ([] if k=="events" else None if k=="last_pred" else False)
        st.rerun()


# ═══════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════
st.markdown("""<div class="app-header"><div>
<div class="logo">🛡 FightSense AI</div>
<div class="sub">Real‑Time Violence Detection · MobileNetV2 + LSTM</div></div></div>""",unsafe_allow_html=True)

# Alert banner
pred = st.session_state.last_pred
if pred and pred["label"]=="FIGHT" and pred["confidence"]>=thresh:
    st.markdown(f"""<div class="alert-fight">⚠️ <strong>FIGHT DETECTED</strong> — Confidence: <strong>{int(pred['confidence']*100)}%</strong></div>""",unsafe_allow_html=True)
    if alarm_on and not st.session_state.alarm_playing:
        st.audio(alarm_bytes,format=alarm_fmt,autoplay=True,loop=True,key=f"alarm_{st.session_state.alarm_ctr}")
        st.session_state.alarm_playing=True; st.session_state.alarm_ctr+=1
elif pred and pred["label"]=="SAFE":
    st.markdown(f"""<div class="alert-safe">✅ <strong>SCENE NORMAL</strong> — Confidence: <strong>{int(pred['confidence']*100)}%</strong></div>""",unsafe_allow_html=True)
    st.session_state.alarm_playing=False


# ═══════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════
t1,t2 = st.tabs(["📷 Live Camera","🎬 Upload Video"])

# ── TAB 1: LIVE CAMERA ──
with t1:
    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="section-label">Camera Feed</div>',unsafe_allow_html=True)
        src = st.selectbox("Source",[0,1,2],label_visibility="collapsed")
        b1,b2 = st.columns(2)
        with b1:
            if st.button("▶ Start",type="primary",use_container_width=True): st.session_state.running=True
        with b2:
            if st.button("⏹ Stop",use_container_width=True): st.session_state.running=False
        vph = st.empty()
    with c2:
        st.markdown('<div class="section-label">Live Prediction</div>',unsafe_allow_html=True)
        pph = st.empty(); bph = st.empty(); eph = st.empty()

    if st.session_state.running:
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            st.error("Cannot open camera"); st.session_state.running=False
        else:
            fi, tp = 0, time.time()
            while st.session_state.running:
                ret, frm = cap.read()
                if not ret: time.sleep(0.1); continue
                fi+=1; st.session_state.frame_count+=1
                if fi%skip==0:
                    pr = predict(frm, st.session_state.buf)
                    st.session_state.last_pred = pr
                    if pr["label"]=="FIGHT": st.session_state.fight_count+=1
                    ev = st.session_state.events
                    if not ev or ev[-1]["label"]!=pr["label"]:
                        ev.append({"ts":time.strftime("%H:%M:%S"),"label":pr["label"],"conf":pr["confidence"]})
                        if len(ev)>50: ev.pop(0)
                pr = st.session_state.last_pred or {"label":"–","fight_prob":0,"safe_prob":0,"confidence":0}
                tn = time.time(); fps = 1/max(tn-tp,1e-6); tp=tn
                st.session_state.fps_hist.append(fps); afps = np.mean(st.session_state.fps_hist)
                disp = overlay(frm,pr,afps)
                vph.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB),channels="RGB",use_container_width=True)
                bc = "badge-danger" if pr["label"]=="FIGHT" else "badge-safe"
                bt = "🔴 FIGHT" if pr["label"]=="FIGHT" else "🟢 SAFE"
                pph.markdown(f"""<div class="metric-card {'danger' if pr['label']=='FIGHT' else 'safe'}">
                <div class="val">{int(pr['confidence']*100)}%</div><div class="label">{pr['label']} Confidence</div></div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;">
                <div class="metric-card accent"><div class="val" style="font-size:1.5rem;">{st.session_state.fight_count}</div><div class="label">Fights</div></div>
                <div class="metric-card accent"><div class="val" style="font-size:1.5rem;">{afps:.1f}</div><div class="label">FPS</div></div></div>""",unsafe_allow_html=True)
                bph.markdown(f"""<div style="margin-top:12px;">
                <div class="conf-bar-wrap"><div class="bar-label"><span>FIGHT</span><span>{int(pr['fight_prob']*100)}%</span></div><div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{int(pr['fight_prob']*100)}%;background:#ff3b3b;"></div></div></div>
                <div class="conf-bar-wrap"><div class="bar-label"><span>SAFE</span><span>{int(pr['safe_prob']*100)}%</span></div><div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{int(pr['safe_prob']*100)}%;background:#00e5a0;"></div></div></div></div>""",unsafe_allow_html=True)
                eph.markdown('<div style="margin-top:12px;"><div class="section-label">Events</div>'+"".join(f"""<div class="event-row {'fight' if e['label']=='FIGHT' else 'safe'}"><span class="event-ts">{e['ts']}</span><span>{'🔴' if e['label']=='FIGHT' else '🟢'}</span><span style="flex:1;font-weight:700;">{e['label']}</span><span style="color:var(--muted);">{int(e['conf']*100)}%</span></div>""" for e in reversed(st.session_state.events[-10:]))+'</div>',unsafe_allow_html=True)
            cap.release()
    else:
        vph.markdown("""<div style="background:var(--card-bg);border:1px dashed var(--border);border-radius:8px;height:380px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;"><div style="font-size:3rem;">📷</div><div style="font-size:0.8rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;">Camera paused</div><div style="font-size:0.68rem;color:var(--muted);">Press ▶ Start</div></div>""",unsafe_allow_html=True)


# ── TAB 2: UPLOAD VIDEO ──
with t2:
    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="section-label">Upload Video</div>',unsafe_allow_html=True)
        up = st.file_uploader("Drag & drop",type=["mp4","avi","mov","mkv"],label_visibility="collapsed")
        if up:
            st.markdown(f"""<div style="background:var(--card-bg);border:1px solid var(--border);border-radius:6px;padding:12px 16px;margin:8px 0;font-size:0.75rem;display:flex;justify-content:space-between;"><span>📂 {up.name}</span><span style="color:var(--muted);">{up.size//1024} KB</span></div>""",unsafe_allow_html=True)
            if st.button("🔍 Analyze",type="primary",use_container_width=True):
                tf_ = tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tf_.write(up.read()); tf_.close()
                cap = cv2.VideoCapture(tf_.name)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25
                vph2 = st.empty(); pph2 = st.empty(); iph2 = st.empty()
                buf2 = deque(maxlen=SEQUENCE_LENGTH)
                fi2, ff2, probs, stamps = 0,0,[],[]
                iph2.markdown('<div class="badge badge-warn">⏳ Analyzing...</div>',unsafe_allow_html=True)
                lastp = {"label":"SAFE","fight_prob":0,"safe_prob":1,"confidence":0}
                while True:
                    ret,frm = cap.read()
                    if not ret: break
                    fi2+=1
                    if fi2%skip==0:
                        lastp = predict(frm,buf2); probs.append(lastp["fight_prob"])
                        if lastp["label"]=="FIGHT":
                            ff2+=1; stamps.append({"ts":f"{int(fi2/fps_vid//60):02d}:{int(fi2/fps_vid%60):02d}","conf":lastp["confidence"]})
                    pph2.progress(fi2/max(total,1),text=f"Frame {fi2}/{total}")
                    if fi2%(skip*5)==0:
                        disp2 = overlay(frm,lastp,fps_vid)
                        vph2.image(cv2.cvtColor(disp2,cv2.COLOR_BGR2RGB),channels="RGB",use_container_width=True)
                cap.release(); os.unlink(tf_.name); pph2.empty()
                st.session_state.frame_count+=fi2; st.session_state.fight_count+=ff2
                inf = len(probs); fpct = ff2/max(inf,1)*100; avgp = np.mean(probs) if probs else 0; mxp = np.max(probs) if probs else 0
                verdict = "FIGHT DETECTED" if fpct>20 else "NO FIGHT"
                iph2.markdown(f'<div class="badge {"badge-danger" if fpct>20 else "badge-safe"}">{verdict}</div>',unsafe_allow_html=True)
                if fpct>20 and alarm_on: st.audio(alarm_bytes,format=alarm_fmt,autoplay=True,loop=True)
                st.session_state["vr"] = {"fpct":fpct,"avgp":avgp,"mxp":mxp,"ff":ff2,"inf":inf,"probs":probs,"stamps":stamps[-20:],"verdict":verdict}
        else:
            st.markdown("""<div style="background:var(--card-bg);border:1px dashed var(--border);border-radius:8px;height:340px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;"><div style="font-size:3rem;">🎬</div><div style="font-size:0.8rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;">Upload a video</div></div>""",unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-label">Results</div>',unsafe_allow_html=True)
        vr = st.session_state.get("vr")
        if vr:
            fp = vr["fpct"]
            st.markdown(f"""<div class="metric-card {'danger' if fp>20 else 'safe'}" style="margin-bottom:12px;"><div class="val">{fp:.1f}%</div><div class="label">Fight Frame Rate</div></div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;">
            <div class="metric-card warn"><div class="val" style="font-size:1.4rem;">{vr['avgp']*100:.1f}%</div><div class="label">Avg Prob</div></div>
            <div class="metric-card danger"><div class="val" style="font-size:1.4rem;">{vr['mxp']*100:.1f}%</div><div class="label">Peak Prob</div></div></div>""",unsafe_allow_html=True)
            if vr["stamps"]:
                st.markdown('<div class="section-label">Fight Events</div>',unsafe_allow_html=True)
                st.markdown("".join(f"""<div class="event-row fight"><span class="event-ts">{e['ts']}</span><span>🔴</span><span style="flex:1;font-weight:700;">FIGHT</span><span style="color:var(--muted);">{int(e['conf']*100)}%</span></div>""" for e in vr["stamps"]),unsafe_allow_html=True)
            st.markdown(f"""<div style="margin-top:16px;padding:12px 16px;background:var(--card-bg);border:1px solid var(--border);border-radius:6px;font-size:0.72rem;color:var(--muted);">
            <div>Frames: <span style="color:var(--text);">{vr['inf']}</span></div>
            <div>Fights: <span style="color:var(--danger);">{vr['ff']}</span></div>
            <div>Verdict: <span style="color:{'var(--danger)' if fp>20 else 'var(--safe)'};">{vr['verdict']}</span></div></div>""",unsafe_allow_html=True)

st.markdown("---")
st.markdown("""<div style="display:flex;justify-content:space-between;padding-bottom:20px;font-size:0.65rem;color:var(--muted);"><span>FightSense AI · MobileNetV2 + LSTM · RWF-2000</span><span>best_fight_model.h5</span></div>""",unsafe_allow_html=True)