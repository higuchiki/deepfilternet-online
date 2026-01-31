import os
import streamlit as st
import torch
import torchaudio
import numpy as np
import time
import tempfile
import subprocess
import threading
import base64
from df.enhance import enhance, init_df, load_audio, save_audio

# モデルの初期化（キャッシュして高速化）
@st.cache_resource
def get_model():
    # メモリ節約のため、Smallモデルを選択肢に入れることも検討できますが、まずは標準V3でチャンク化
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
    /* 全体の背景とフォント */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp {
        background-color: #000000;
        color: #ededed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* メインコンテナの最大幅を制限して中央寄せ */
    .main .block-container {
        max-width: 1000px;
        padding-top: 4rem;
    }

    /* サイドバー */
    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #333333;
    }
    
    /* ヘッダー・タイトル */
    .main-title {
        font-weight: 800;
        font-size: 2.5rem !important;
        letter-spacing: -0.05em;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .sub-title {
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* アップローダー */
    .stFileUploader {
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        background-color: #0a0a0a !important;
        padding: 2rem !important;
    }
    .stFileUploader section {
        background-color: transparent !important;
    }
    
    /* ボタン */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: 1px solid #ffffff !important;
        height: 2.8rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    /* オーディオプレイヤー・カード */
    .audio-card {
        background: #0a0a0a;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #333333;
        margin-bottom: 1rem;
    }
    .audio-card b {
        color: #888888;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: block;
        margin-bottom: 0.8rem;
    }
    
    /* ダウンロードボタンのカスタマイズ */
    .stDownloadButton {
        text-align: center;
    }
    .stDownloadButton > button {
        width: auto !important;
        min-width: 300px !important;
        padding: 0.8rem 2rem !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%) !important;
        color: #000000 !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(255,255,255,0.1), 0 10px 30px rgba(0,0,0,0.5) !important;
        margin: 2.5rem auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(255,255,255,0.15), 0 20px 50px rgba(0,0,0,0.6) !important;
        background: #ffffff !important;
    }

    /* プログレスバー */
    .stProgress > div > div > div > div {
        background-color: #ffffff;
    }

    /* 処理中のローダーアニメーション */
    .processing-container {
        text-align: center;
        padding: 2rem;
        background: #0a0a0a;
        border-radius: 8px;
        border: 1px solid #333333;
        margin: 1rem 0;
    }
    .loader {
        border: 3px solid #333333;
        border-top: 3px solid #ffffff;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .timer-text {
        font-family: monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# メインコンテンツ
st.markdown('<h1 class="main-title">DeepFilterNet AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">High-performance noise suppression for professional audio.</p>', unsafe_allow_html=True)

# モデルの初期化（エラー時のみ表示）
try:
    model, df_state = get_model()
except Exception as e:
    st.error(f"AI Model Error: {e}")
    st.stop()

# メインレイアウト
st.subheader("1. Upload Audio")
uploaded_file = st.file_uploader("", type=["wav", "m4a", "mp3", "aac"])

if uploaded_file:
    st.markdown("---")
    st.subheader("2. Configuration")
    atten_lim = st.slider(
        "Attenuation Limit (dB)", 
        min_value=0, max_value=100, value=0,
        help="Lower values mean stronger noise reduction. 0dB is recommended for maximum suppression."
    )
    
    if st.button("Enhance Audio"):
        if 'processed_data' in st.session_state:
            del st.session_state['processed_data']
            
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_path = os.path.join(tmpdirname, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            processing_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            start_time = time.time()
            
            st.components.v1.html(f"""
                <div id="timer-root"></div>
                <script>
                let start = Date.now();
                const root = document.getElementById('timer-root');
                // Streamlitの親要素にスタイルを適用するためのハック
                window.parent.document.querySelectorAll('.timer-text').forEach(el => {{
                    setInterval(() => {{
                        let elapsed = (Date.now() - start) / 1000;
                        el.innerText = elapsed.toFixed(1) + 's';
                    }}, 100);
                }});
                </script>
            """, height=0)

            processing_placeholder.markdown(f"""
                <div class="processing-container">
                    <div class="loader"></div>
                    <div style="color: #888; font-size: 0.9rem; margin-bottom: 0.5rem;">AI Processing (Chunked Mode)...</div>
                    <div class="timer-text">0.0s</div>
                </div>
            """, unsafe_allow_html=True)

            try:
                # 1. 読み込みと変換
                progress_bar.progress(10)
                base, ext = os.path.splitext(input_path)
                load_path = input_path
                if ext.lower() in [".m4a", ".mp3", ".mp4", ".aac"]:
                    temp_wav = os.path.join(tmpdirname, "temp_conv.wav")
                    subprocess.run(["ffmpeg", "-y", "-i", input_path, temp_wav], check=True, capture_output=True)
                    load_path = temp_wav
                
                audio, info = load_audio(load_path, sr=df_state.sr())
                
                # 2. チャンク処理によるノイズ除去
                # メモリ節約のため、音声を分割して処理
                chunk_size_sec = 30 # 30秒ごとに処理
                sr = df_state.sr()
                chunk_size_samples = chunk_size_sec * sr
                total_samples = audio.shape[1]
                
                enhanced_chunks = []
                num_chunks = int(np.ceil(total_samples / chunk_size_samples))
                
                proc_start = time.time()
                for i in range(num_chunks):
                    start_sample = i * chunk_size_samples
                    end_sample = min((i + 1) * chunk_size_samples, total_samples)
                    
                    audio_chunk = audio[:, start_sample:end_sample]
                    
                    # AI処理
                    enhanced_chunk = enhance(model, df_state, audio_chunk, atten_lim_db=atten_lim)
                    enhanced_chunks.append(enhanced_chunk)
                    
                    # 進捗更新 (10%から90%の間で動かす)
                    p = 10 + int((i + 1) / num_chunks * 80)
                    progress_bar.progress(p)
                
                # 結合
                enhanced = torch.cat(enhanced_chunks, dim=1)
                proc_duration = time.time() - proc_start
                
                # 3. 保存
                output_path = os.path.join(tmpdirname, "enhanced.wav")
                save_audio(output_path, enhanced, sr=df_state.sr())
                
                progress_bar.progress(100)
                
                with open(output_path, "rb") as f:
                    audio_bytes = f.read()
                
                st.session_state['processed_data'] = {
                    'input_file': uploaded_file.getvalue(),
                    'output_bytes': audio_bytes,
                    'filename': uploaded_file.name,
                    'duration': proc_duration
                }
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
                processing_placeholder.empty()

    st.markdown("---")
    st.subheader("3. Results")
    
    if 'processed_data' in st.session_state:
        res = st.session_state['processed_data']
        st.success(f"🎉 Success: Processed in {res['duration']:.1f}s")
        
        input_b64 = base64.b64encode(res['input_file']).decode()
        output_b64 = base64.b64encode(res['output_bytes']).decode()
        
        st.components.v1.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
            body {{ background-color: transparent; margin: 0; font-family: 'Inter', sans-serif; color: white; }}
            .player-container {{ background: #0a0a0a; border: 1px solid #333333; border-radius: 12px; padding: 1.5rem; }}
            .controls {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }}
            .play-btn {{ background: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer; display: flex; align-items: center; justify-content: center; }}
            .play-btn svg {{ fill: black; width: 20px; height: 20px; }}
            .seek-bar {{ flex-grow: 1; cursor: pointer; accent-color: white; }}
            .toggle-container {{ display: flex; background: #1a1a1a; border-radius: 8px; padding: 4px; border: 1px solid #333; width: fit-content; margin: 0 auto; }}
            .toggle-btn {{ padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; transition: all 0.2s; border: none; background: transparent; color: #888; }}
            .toggle-btn.active {{ background: #333; color: white; }}
            .time-display {{ font-family: monospace; font-size: 0.85rem; color: #888; min-width: 80px; }}
            </style>
            <div class="player-container">
                <div class="controls">
                    <button id="playBtn" class="play-btn"><svg id="playIcon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg><svg id="pauseIcon" style="display:none" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg></button>
                    <div class="time-display" id="currentTime">0:00</div>
                    <input type="range" id="seekBar" class="seek-bar" value="0" step="0.1">
                    <div class="time-display" id="totalTime">0:00</div>
                </div>
                <div class="toggle-container">
                    <button id="origBtn" class="toggle-btn">Original Source</button>
                    <button id="enhBtn" class="toggle-btn active">AI Enhanced</button>
                </div>
            </div>
            <audio id="audioOrig" src="data:audio/wav;base64,{input_b64}" preload="auto"></audio>
            <audio id="audioEnh" src="data:audio/wav;base64,{output_b64}" preload="auto"></audio>
            <script>
            const audioOrig = document.getElementById('audioOrig');
            const audioEnh = document.getElementById('audioEnh');
            const playBtn = document.getElementById('playBtn');
            const playIcon = document.getElementById('playIcon');
            const pauseIcon = document.getElementById('pauseIcon');
            const seekBar = document.getElementById('seekBar');
            const currentTime = document.getElementById('currentTime');
            const totalTime = document.getElementById('totalTime');
            const origBtn = document.getElementById('origBtn');
            const enhBtn = document.getElementById('enhBtn');
            let isPlaying = false;
            let currentSource = 'enhanced';
            audioOrig.muted = true;
            audioEnh.muted = false;
            function formatTime(secs) {{ const m = Math.floor(secs / 60); const s = Math.floor(secs % 60); return m + ":" + (s < 10 ? "0" + s : s); }}
            async function togglePlay() {{
                if (isPlaying) {{ audioOrig.pause(); audioEnh.pause(); playIcon.style.display = 'block'; pauseIcon.style.display = 'none'; }}
                else {{ await Promise.all([audioOrig.play(), audioEnh.play()]); playIcon.style.display = 'none'; pauseIcon.style.display = 'block'; }}
                isPlaying = !isPlaying;
            }}
            playBtn.onclick = togglePlay;
            origBtn.onclick = () => {{ currentSource = 'original'; origBtn.classList.add('active'); enhBtn.classList.remove('active'); audioOrig.muted = false; audioEnh.muted = true; audioOrig.currentTime = audioEnh.currentTime; }};
            enhBtn.onclick = () => {{ currentSource = 'enhanced'; enhBtn.classList.add('active'); origBtn.classList.remove('active'); audioEnh.muted = false; audioOrig.muted = true; audioEnh.currentTime = audioOrig.currentTime; }};
            audioEnh.onloadedmetadata = () => {{ totalTime.innerText = formatTime(audioEnh.duration); seekBar.max = audioEnh.duration; }};
            audioEnh.ontimeupdate = () => {{ if (!isDragging) {{ seekBar.value = audioEnh.currentTime; currentTime.innerText = formatTime(audioEnh.currentTime); }} }};
            let isDragging = false;
            seekBar.onmousedown = () => {{ isDragging = true; }};
            seekBar.onmouseup = () => {{ isDragging = false; }};
            seekBar.oninput = () => {{ const val = seekBar.value; audioEnh.currentTime = val; audioOrig.currentTime = val; }};
            audioEnh.onended = () => {{ isPlaying = false; playIcon.style.display = 'block'; pauseIcon.style.display = 'none'; audioOrig.pause(); audioOrig.currentTime = 0; audioEnh.currentTime = 0; }};
            </script>
        """, height=180)
        
        st.download_button(
            label="📥 Download Enhanced Audio",
            data=res['output_bytes'],
            file_name=f"{os.path.splitext(res['filename'])[0]}_enhanced.wav",
            mime="audio/wav"
        )
    else:
        st.info("Upload audio and click 'Enhance Audio' to see results.")

# ページ下段に情報を隔離
st.markdown("<br><br><br><br>", unsafe_allow_html=True)
st.divider()

# 原作者へのクレジット
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #888888;">
        <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">Powered by the incredible work of</p>
        <a href="https://github.com/Rikorose/DeepFilterNet" target="_blank" style="color: #ffffff; text-decoration: none; font-weight: 600; font-size: 1.1rem;">
            Hendrik Schröter (Rikorose)
        </a>
        <p style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.6;">
            Thank you for making high-quality noise suppression accessible to everyone.
        </p>
    </div>
""", unsafe_allow_html=True)

with st.expander("View Documentation & Technical Specs"):
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.markdown("### Documentation")
        st.markdown("""
        **Deep Filtering**
        AI-powered frequency separation. Unlike traditional gates, it preserves speech quality while removing background noise.

        **Performance**
        Optimized Rust engine for near real-time processing on standard CPUs.

        **Privacy**
        Files are processed in-memory and never stored on disk beyond the session.
        """)
    
    with exp_col2:
        st.markdown("### Technical Specs")
        st.code("""
Sampling Rate: 48kHz
Model: DeepFilterNet V3
Backend: PyTorch / Rust
        """)
