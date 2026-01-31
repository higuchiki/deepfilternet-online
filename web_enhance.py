import os
import streamlit as st
import torch
import torchaudio
import numpy as np
import time
import tempfile
import subprocess
import base64
from df.enhance import enhance, init_df, load_audio, save_audio

# モデルの初期化（キャッシュして高速化）
@st.cache_resource
def get_model():
    model, df_state, _ = init_df()
    return model, df_state

# ページ設定
st.set_page_config(
    page_title="DeepFilterNet AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Next.js Docs 風のダークモード・ミニマルデザイン
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background-color: #000000; color: #ededed; font-family: 'Inter', sans-serif; }
    .main .block-container { max-width: 800px; padding-top: 4rem; }
    .main-title { font-weight: 800; font-size: 2.5rem !important; letter-spacing: -0.05em; margin-bottom: 0.5rem; color: #ffffff; text-align: center; }
    .sub-title { color: #888888; font-size: 1.1rem; margin-bottom: 3rem; text-align: center; }
    .stFileUploader { border: 1px solid #333333 !important; border-radius: 8px !important; background-color: #0a0a0a !important; padding: 1rem !important; }
    .stButton > button { background-color: #ffffff !important; color: #000000 !important; border-radius: 6px !important; font-weight: 600 !important; height: 3rem; width: 100%; transition: all 0.2s ease; border: none; }
    .audio-card { background: #0a0a0a; padding: 1.2rem; border-radius: 8px; border: 1px solid #333333; margin-bottom: 1rem; }
    .audio-card b { color: #4A90E2; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
    .stDownloadButton > button { width: auto !important; min-width: 300px !important; padding: 0.8rem 2rem !important; background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%) !important; color: #000000 !important; border-radius: 12px !important; font-weight: 700 !important; margin: 2rem auto !important; display: flex !important; align-items: center !important; justify-content: center !important; transition: all 0.3s ease !important; border: none; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">DeepFilterNet AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Professional noise suppression powered by AI.</p>', unsafe_allow_html=True)

try:
    model, df_state = get_model()
except Exception as e:
    st.error(f"AI Model Error: {e}")
    st.stop()

st.subheader("1. Upload Audio")
uploaded_file = st.file_uploader("Select WAV, M4A, MP3, or AAC file", type=["wav", "m4a", "mp3", "aac"])

if uploaded_file:
    st.markdown("---")
    st.subheader("2. Configuration")
    atten_lim = st.slider("Noise Reduction Limit (dB)", 0, 100, 0, help="0dB = Maximum reduction")
    
    if st.button("✨ Enhance Audio"):
        with st.status("AI is working...", expanded=True) as status:
            with tempfile.TemporaryDirectory() as tmpdirname:
                input_path = os.path.join(tmpdirname, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    st.write("Preparing audio...")
                    load_path = input_path
                    if not input_path.lower().endswith(".wav"):
                        temp_wav = os.path.join(tmpdirname, "temp.wav")
                        subprocess.run(["ffmpeg", "-y", "-i", input_path, temp_wav], check=True, capture_output=True)
                        load_path = temp_wav
                    
                    audio, _ = load_audio(load_path, sr=df_state.sr())
                    
                    st.write("AI Processing (Chunked Mode)...")
                    chunk_size = 30 * df_state.sr()
                    total = audio.shape[1]
                    chunks = []
                    
                    proc_start = time.time()
                    progress_bar = st.progress(0)
                    for i in range(0, total, chunk_size):
                        chunk = audio[:, i:i+chunk_size]
                        enhanced_chunk = enhance(model, df_state, chunk, atten_lim_db=atten_lim)
                        chunks.append(enhanced_chunk)
                        progress_bar.progress(min(int(i/total*100), 100))
                    
                    enhanced = torch.cat(chunks, dim=1)
                    proc_duration = time.time() - proc_start
                    
                    st.write("Generating results...")
                    output_path = os.path.join(tmpdirname, "enhanced.wav")
                    save_audio(output_path, enhanced, sr=df_state.sr())
                    with open(output_path, "rb") as f:
                        audio_bytes = f.read()
                    
                    st.session_state['processed_data'] = {
                        'input': uploaded_file.getvalue(),
                        'output': audio_bytes,
                        'name': uploaded_file.name,
                        'time': proc_duration
                    }
                    status.update(label=f"✅ Done! Processed in {proc_duration:.1f}s", state="complete")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    status.update(label="❌ Error occurred", state="error")

    st.markdown("---")
    st.subheader("3. Results")
    if 'processed_data' in st.session_state:
        res = st.session_state['processed_data']
        
        in_b64 = base64.b64encode(res['input']).decode()
        out_b64 = base64.b64encode(res['output']).decode()
        
        st.components.v1.html(f"""
            <style>
            body {{ background: transparent; margin: 0; font-family: sans-serif; color: white; }}
            .player {{ background: #0a0a0a; border: 1px solid #333; border-radius: 12px; padding: 1.5rem; }}
            .ctrl {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }}
            .p-btn {{ background: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer; font-size: 1.2rem; }}
            .sk {{ flex-grow: 1; accent-color: white; }}
            .tgl-c {{ display: flex; background: #1a1a1a; border-radius: 8px; padding: 4px; border: 1px solid #333; width: fit-content; margin: 0 auto; }}
            .tgl {{ padding: 8px 20px; border-radius: 6px; cursor: pointer; border: none; background: transparent; color: #888; font-weight: 600; }}
            .tgl.active {{ background: #333; color: white; }}
            </style>
            <div class="player">
                <div class="ctrl">
                    <button id="p" class="p-btn">▶</button>
                    <span id="ct" style="font-family:monospace; color:#888;">0:00</span>
                    <input type="range" id="s" class="sk" value="0" step="0.1">
                    <span id="tt" style="font-family:monospace; color:#888;">0:00</span>
                </div>
                <div class="tgl-c">
                    <button id="b1" class="tgl">Original</button>
                    <button id="b2" class="tgl active">AI Enhanced</button>
                </div>
            </div>
            <audio id="a1" src="data:audio/wav;base64,{in_b64}"></audio>
            <audio id="a2" src="data:audio/wav;base64,{out_b64}"></audio>
            <script>
            const a1=document.getElementById('a1'), a2=document.getElementById('a2'), p=document.getElementById('p'), s=document.getElementById('s'), ct=document.getElementById('ct'), tt=document.getElementById('tt'), b1=document.getElementById('b1'), b2=document.getElementById('b2');
            let playing=false; a1.muted=true; a2.muted=false;
            const fmt=t=>{{const m=Math.floor(t/60),s=Math.floor(t%60); return m+":"+(s<10?"0"+s:s)}};
            p.onclick=async()=>{{ if(playing){{a1.pause();a2.pause();p.innerText='▶'}}else{{await Promise.all([a1.play(),a2.play()]);p.innerText='||'}} playing=!playing }};
            b1.onclick=()=>{{ b1.classList.add('active'); b2.classList.remove('active'); a1.muted=false; a2.muted=true; a1.currentTime=a2.currentTime; }};
            b2.onclick=()=>{{ b2.classList.add('active'); b1.classList.remove('active'); a2.muted=false; a1.muted=true; a2.currentTime=a1.currentTime; }};
            a2.onloadedmetadata=()=>{{ tt.innerText=fmt(a2.duration); s.max=a2.duration; }};
            a2.ontimeupdate=()=>{{ s.value=a2.currentTime; ct.innerText=fmt(a2.currentTime); }};
            s.oninput=()=>{{ a1.currentTime=a2.currentTime=s.value; }};
            </script>
        """, height=160)
        
        st.download_button("📥 Download Enhanced Audio", res['output'], f"{os.path.splitext(res['name'])[0]}_enhanced.wav", "audio/wav")
    else:
        st.info("Upload and click 'Enhance Audio' to see results.")

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.divider()
st.markdown('<div style="text-align:center;color:#888;font-size:0.9rem;">Powered by <a href="https://github.com/Rikorose/DeepFilterNet" style="color:#fff;text-decoration:none;font-weight:600;">Hendrik Schröter (Rikorose)</a></div>', unsafe_allow_html=True)
