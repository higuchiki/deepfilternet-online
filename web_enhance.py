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
    .main .block-container { max-width: 1000px; padding-top: 4rem; }
    .main-title { font-weight: 800; font-size: 2.5rem !important; letter-spacing: -0.05em; margin-bottom: 0.5rem; color: #ffffff; }
    .sub-title { color: #888888; font-size: 1.1rem; margin-bottom: 3rem; }
    .stFileUploader { border: 1px solid #333333 !important; border-radius: 8px !important; background-color: #0a0a0a !important; padding: 2rem !important; }
    .stButton > button { background-color: #ffffff !important; color: #000000 !important; border-radius: 6px !important; font-weight: 600 !important; height: 2.8rem; width: 100%; transition: all 0.2s ease; }
    .audio-card { background: #0a0a0a; padding: 1.2rem; border-radius: 8px; border: 1px solid #333333; margin-bottom: 1rem; }
    .audio-card b { color: #4A90E2; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
    .stDownloadButton > button { width: auto !important; min-width: 300px !important; padding: 0.8rem 2rem !important; background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%) !important; color: #000000 !important; border-radius: 12px !important; font-weight: 700 !important; margin: 2.5rem auto !important; display: flex !important; align-items: center !important; justify-content: center !important; transition: all 0.3s ease !important; }
    .processing-container { text-align: center; padding: 2rem; background: #0a0a0a; border-radius: 8px; border: 1px solid #333333; margin: 1rem 0; }
    .loader { border: 3px solid #333333; border-top: 3px solid #ffffff; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 1rem; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .timer-text { font-family: monospace; font-size: 1.5rem; font-weight: 600; color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">DeepFilterNet AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">High-performance noise suppression for professional audio.</p>', unsafe_allow_html=True)

try:
    model, df_state = get_model()
except Exception as e:
    st.error(f"AI Model Error: {e}")
    st.stop()

st.subheader("1. Upload Audio")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "m4a", "mp3", "aac"], label_visibility="collapsed")

if uploaded_file:
    st.markdown("---")
    st.subheader("2. Configuration")
    atten_lim = st.slider("Attenuation Limit (dB)", 0, 100, 0)
    
    if st.button("Enhance Audio"):
        if 'processed_data' in st.session_state:
            del st.session_state['processed_data']
            
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_path = os.path.join(tmpdirname, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            placeholder = st.empty()
            progress_bar = st.progress(0)
            start_time = time.time()
            
            # タイマー表示 (iframe)
            st.components.v1.html(f"""
                <script>
                let start = Date.now();
                setInterval(() => {{
                    let elapsed = (Date.now() - start) / 1000;
                    window.parent.document.querySelectorAll('.timer-text').forEach(el => {{
                        el.innerText = elapsed.toFixed(1) + 's';
                    }});
                }}, 100);
                </script>
            """, height=0)

            placeholder.markdown(f"""
                <div class="processing-container">
                    <div class="loader"></div>
                    <div style="color: #888; font-size: 0.9rem; margin-bottom: 0.5rem;">AI Processing (Chunked Mode)...</div>
                    <div class="timer-text">0.0s</div>
                </div>
            """, unsafe_allow_html=True)

            try:
                # 1. 変換
                progress_bar.progress(10)
                load_path = input_path
                if not input_path.lower().endswith(".wav"):
                    temp_wav = os.path.join(tmpdirname, "temp.wav")
                    subprocess.run(["ffmpeg", "-y", "-i", input_path, temp_wav], check=True, capture_output=True)
                    load_path = temp_wav
                
                audio, _ = load_audio(load_path, sr=df_state.sr())
                
                # 2. チャンク処理
                chunk_size = 30 * df_state.sr()
                total = audio.shape[1]
                chunks = []
                
                proc_start = time.time()
                for i in range(0, total, chunk_size):
                    chunk = audio[:, i:i+chunk_size]
                    enhanced_chunk = enhance(model, df_state, chunk, atten_lim_db=atten_lim)
                    chunks.append(enhanced_chunk)
                    progress_bar.progress(min(10 + int(i/total*80), 90))
                
                enhanced = torch.cat(chunks, dim=1)
                proc_duration = time.time() - proc_start
                
                # 3. 保存
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
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
                placeholder.empty()

    st.markdown("---")
    st.subheader("3. Results")
    if 'processed_data' in st.session_state:
        res = st.session_state['processed_data']
        st.success(f"🎉 Success: Processed in {res['time']:.1f}s")
        
        in_b64 = base64.b64encode(res['input']).decode()
        out_b64 = base64.b64encode(res['output']).decode()
        
        st.components.v1.html(f"""
            <style>
            body {{ background: transparent; margin: 0; font-family: sans-serif; color: white; }}
            .player {{ background: #0a0a0a; border: 1px solid #333; border-radius: 12px; padding: 1.5rem; }}
            .ctrl {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }}
            .p-btn {{ background: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer; }}
            .sk {{ flex-grow: 1; accent-color: white; }}
            .tgl-c {{ display: flex; background: #1a1a1a; border-radius: 8px; padding: 4px; border: 1px solid #333; width: fit-content; margin: 0 auto; }}
            .tgl {{ padding: 6px 16px; border-radius: 6px; cursor: pointer; border: none; background: transparent; color: #888; }}
            .tgl.active {{ background: #333; color: white; }}
            </style>
            <div class="player">
                <div class="ctrl">
                    <button id="p" class="p-btn">▶</button>
                    <span id="ct">0:00</span>
                    <input type="range" id="s" class="sk" value="0" step="0.1">
                    <span id="tt">0:00</span>
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
            let playing=false, src='enh'; a1.muted=true; a2.muted=false;
            const fmt=t=>{{const m=Math.floor(t/60),s=Math.floor(t%60); return m+":"+(s<10?"0"+s:s)}};
            p.onclick=async()=>{{ if(playing){{a1.pause();a2.pause();p.innerText='▶'}}else{{await Promise.all([a1.play(),a2.play()]);p.innerText='||'}} playing=!playing }};
            b1.onclick=()=>{{ src='orig'; b1.classList.add('active'); b2.classList.remove('active'); a1.muted=false; a2.muted=true; a1.currentTime=a2.currentTime; }};
            b2.onclick=()=>{{ src='enh'; b2.classList.add('active'); b1.classList.remove('active'); a2.muted=false; a1.muted=true; a2.currentTime=a1.currentTime; }};
            a2.onloadedmetadata=()=>{{ tt.innerText=fmt(a2.duration); s.max=a2.duration; }};
            a2.ontimeupdate=()=>{{ s.value=a2.currentTime; ct.innerText=fmt(a2.currentTime); }};
            s.oninput=()=>{{ a1.currentTime=a2.currentTime=s.value; }};
            </script>
        """, height=160)
        
        st.download_button("📥 Download Enhanced Audio", res['output'], f"{os.path.splitext(res['name'])[0]}_enhanced.wav", "audio/wav")
    else:
        st.info("Upload and click 'Enhance Audio'.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown('<div style="text-align:center;color:#888;">Powered by <a href="https://github.com/Rikorose/DeepFilterNet" style="color:#fff;text-decoration:none;">Hendrik Schröter (Rikorose)</a></div>', unsafe_allow_html=True)
