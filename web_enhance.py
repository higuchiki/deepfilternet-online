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

# モデルの初期化
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

# テキスト辞書（日本語固定）
T = {
    'title': 'DeepFilterNet AI 音声ノイズ除去',
    'subtitle': 'AI技術を駆使した、プロフェッショナル仕様のノイズ除去ツール。',
    'step1': '1. 音声をアップロード',
    'uploader_label': 'WAV, M4A, MP3, AAC ファイルを選択してください',
    'step2': '2. 設定',
    'atten_label': 'ノイズ除去の強度制限 (dB)',
    'atten_help': '0dBに近いほど強力にノイズを消します。声が不自然な場合のみ値を大きくしてください。',
    'btn_enhance': '✨ ノイズを除去する',
    'status_preparing': '音声を準備中...',
    'status_processing': 'AIがノイズを除去中 (分割処理モード)...',
    'status_saving': '結果を生成中...',
    'status_done': '完了！ 処理時間: {duration:.1f}秒',
    'step3': '3. 処理結果',
    'success_msg': '🎉 成功: {duration:.1f}秒で処理が完了しました',
    'input_label': '元の音源',
    'output_label': 'AI除去後',
    'btn_download': '📥 除去済み音声をダウンロード',
    'info_msg': 'ファイルをアップロードして「ノイズを除去する」をクリックしてください。',
    'powered_by': 'Powered by',
}

# CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background-color: #000000; color: #ededed; font-family: 'Inter', sans-serif; }
    .main .block-container { max-width: 900px; padding-top: 4rem; }
    .main-title { font-weight: 800; font-size: 2.5rem !important; letter-spacing: -0.05em; margin-bottom: 0.5rem; color: #ffffff; text-align: center; }
    .sub-title { color: #888888; font-size: 1.1rem; margin-bottom: 3rem; text-align: center; }
    .stFileUploader { border: 1px solid #333333 !important; border-radius: 8px !important; background-color: #0a0a0a !important; padding: 1rem !important; }
    .stButton > button { background-color: #ffffff !important; color: #000000 !important; border-radius: 6px !important; font-weight: 600 !important; height: 3rem; width: 100%; transition: all 0.2s ease; border: none; }
    .audio-card { background: #0a0a0a; padding: 1.2rem; border-radius: 8px; border: 1px solid #333333; margin-bottom: 1rem; }
    .audio-card b { color: #4A90E2; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
    .stDownloadButton > button { width: auto !important; min-width: 300px !important; padding: 0.8rem 2rem !important; background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%) !important; color: #000000 !important; border-radius: 12px !important; font-weight: 700 !important; margin: 2.5rem auto !important; display: flex !important; align-items: center !important; justify-content: center !important; transition: all 0.3s ease !important; border: none; }
    
    /* Streamlit標準要素の非表示 */
    #MainMenu, footer, header, div[data-testid="stDecoration"], div[data-testid="stHeader"] {
        visibility: hidden;
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(f'<h1 class="main-title">{T["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{T["subtitle"]}</p>', unsafe_allow_html=True)

try:
    model, df_state = get_model()
except Exception as e:
    st.error(f"AI Model Error: {e}")
    st.stop()

st.subheader(T['step1'])
col_up1, col_up2, col_up3 = st.columns([1, 4, 1])
with col_up2:
    uploaded_file = st.file_uploader(T['uploader_label'], type=["wav", "m4a", "mp3", "aac"], label_visibility="collapsed")

if uploaded_file:
    st.markdown("---")
    st.subheader(T['step2'])
    col_conf1, col_conf2, col_conf3 = st.columns([1, 4, 1])
    with col_conf2:
        atten_lim = st.slider(T['atten_label'], 0, 100, 0, help=T['atten_help'])
        
    if st.button(T['btn_enhance']):
        with st.status(T['status_processing'], expanded=True) as status:
            with tempfile.TemporaryDirectory() as tmpdirname:
                input_path = os.path.join(tmpdirname, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    st.write(T['status_preparing'])
                    load_path = input_path
                    if not input_path.lower().endswith(".wav"):
                        temp_wav = os.path.join(tmpdirname, "temp.wav")
                        subprocess.run(["ffmpeg", "-y", "-i", input_path, temp_wav], check=True, capture_output=True)
                        load_path = temp_wav
                    
                    audio, _ = load_audio(load_path, sr=df_state.sr())
                    
                    st.write(T['status_processing'])
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
                    
                    st.write(T['status_saving'])
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
                    status.update(label=T['status_done'].format(duration=proc_duration), state="complete")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    status.update(label="❌ Error", state="error")

    st.markdown("---")
    st.subheader(T['step3'])
    if 'processed_data' in st.session_state:
        res = st.session_state['processed_data']
        st.success(T['success_msg'].format(duration=res['time']))
        
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
                    <button id="b1" class="tgl">{T['input_label']}</button>
                    <button id="b2" class="tgl active">{T['output_label']}</button>
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
        
        st.download_button(T['btn_download'], res['output'], f"{os.path.splitext(res['name'])[0]}_enhanced.wav", "audio/wav")
    else:
        st.info(T['info_msg'])

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(f'<div style="text-align:center;color:#888;font-size:0.9rem;">{T["powered_by"]} <a href="https://github.com/Rikorose/DeepFilterNet" style="color:#fff;text-decoration:none;font-weight:600;">Hendrik Schröter (Rikorose)</a></div>', unsafe_allow_html=True)
