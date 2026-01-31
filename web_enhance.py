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
    page_title="ClearVoice AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 言語設定（日本語固定）
if 'lang' not in st.session_state:
    st.session_state.lang = 'JP'

# テキスト辞書
T = {
    'title': 'DeepFilterNet ブラウザ版',
    'subtitle': 'AIノイズ処理ライブラリDeepFilterNetを使ったノイズ処理Webアプリ',
    'description': 'AIが、あなたの音声から「雑音」だけを魔法のように消し去ります。',
    'step1': '1. 音源をアップロード',
    'uploader_label': 'WAV, M4A, MP3, AAC ファイルを選択してください',
    'step2': '2. 除去強度の設定',
    'atten_label': 'ノイズ除去の制限 (dB)',
    'atten_help': '0dBに近いほど強力にノイズを消します。声が不自然な場合のみ値を大きくしてください。',
    'btn_enhance': '✨ クリアな音声を生成する',
    'status_preparing': '音声を準備中...',
    'status_processing': 'AIがノイズを解析・除去しています...',
    'status_saving': '結果を生成中...',
    'status_done': '完了！ 処理時間: {duration:.1f}秒',
    'step3': '3. 処理結果',
    'success_msg': '🎉 成功: {duration:.1f}秒でクリアな音声が完成しました',
    'input_label': '元の音源',
    'output_label': 'AI除去後',
    'btn_download': '📥 クリアな音声をダウンロード',
    'info_msg': 'ファイルをアップロードして「クリアな音声を生成する」をクリックしてください。',
    'powered_by': 'Powered by',
}

# CSS: Next.js Docs (Vercel) スタイル
st.markdown("""
    <style>
    /* Vercel / Next.js Docs フォントと背景 */
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@100..900&family=Geist+Mono:wght@100..900&family=Noto+Sans+JP:wght@100..900&display=swap');
    
    :root {
        --background: #000000;
        --foreground: #ededed;
        --muted: #888888;
        --border: #333333;
        --accent: #ffffff;
    }

    .stApp {
        background-color: var(--background);
        color: var(--foreground);
        font-family: 'Geist', 'Noto Sans JP', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* メインコンテナを左寄せに */
    .main .block-container {
        max-width: 1000px;
        margin-left: 0 !important;
        margin-right: auto !important;
        padding-left: 5rem;
        padding-right: 2rem;
        padding-top: 4rem;
    }

    /* タイトルとサブタイトル */
    .main-title {
        font-family: 'Geist', 'Noto Sans JP', sans-serif;
        font-weight: 700;
        font-size: 2.25rem !important;
        letter-spacing: -0.04em;
        margin-bottom: 0.75rem;
        color: #ffffff;
        text-align: left;
    }
    .sub-title {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 3rem;
        text-align: left;
        max-width: 600px;
        line-height: 1.6;
        font-weight: 400;
    }

    /* セクション見出し */
    h2, h3, .stSubheader {
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        text-align: left !important;
    }

    /* アップローダー */
    .stFileUploader {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        background-color: #0a0a0a !important;
        padding: 1.5rem !important;
        max-width: 600px;
    }
    .stFileUploader section {
        background-color: transparent !important;
    }
    
    /* ボタン (Vercelスタイル) */
    .stButton > button {
        background-color: var(--accent) !important;
        color: #000000 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        height: 2.5rem !important;
        width: auto !important;
        min-width: 140px;
        padding: 0 1.5rem !important;
        transition: opacity 0.2s ease;
        border: none !important;
        margin-left: 0 !important;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }

    /* スライダー */
    .stSlider {
        max-width: 600px;
    }

    /* オーディオカード */
    .audio-card {
        background: #0a0a0a;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
        max-width: 450px;
    }
    .audio-card b {
        color: var(--muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: block;
        margin-bottom: 0.8rem;
    }
    
    /* ダウンロードボタン (Vercelスタイル) */
    .stDownloadButton > button {
        width: auto !important;
        min-width: 200px !important;
        padding: 0.6rem 1.5rem !important;
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin-top: 1rem !important;
        transition: background 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #111111 !important;
        border-color: #555555 !important;
    }

    /* Streamlit要素の非表示 */
    #MainMenu, footer, header, div[data-testid="stDecoration"], div[data-testid="stHeader"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# メインコンテンツ
st.markdown(f'<h1 class="main-title">{T["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title" style="margin-bottom: 0.5rem;">{T["subtitle"]}</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color: #666; font-size: 1rem; margin-bottom: 3rem; line-height: 1.6;">{T["description"]}</p>', unsafe_allow_html=True)

try:
    model, df_state = get_model()
except Exception as e:
    st.error(f"AI Model Error: {e}")
    st.stop()

# ステップ1
st.subheader(T['step1'])
col_up1, col_up2 = st.columns([2, 1])
with col_up1:
    uploaded_file = st.file_uploader(T['uploader_label'], type=["wav", "m4a", "mp3", "aac"], label_visibility="collapsed")

if uploaded_file:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(T['step2'])
    col_conf1, col_conf2 = st.columns([2, 1])
    with col_conf1:
        atten_lim = st.slider(T['atten_label'], 0, 100, 0, help=T['atten_help'])
        
        if st.button(T['btn_enhance']):
            if 'processed_data' in st.session_state:
                del st.session_state['processed_data']
                
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
                        p_bar = st.progress(0)
                        for i in range(0, total, chunk_size):
                            chunk = audio[:, i:i+chunk_size]
                            enhanced_chunk = enhance(model, df_state, chunk, atten_lim_db=atten_lim)
                            chunks.append(enhanced_chunk)
                            p_bar.progress(min(int(i/total*100), 100))
                        
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(T['step3'])
    if 'processed_data' in st.session_state:
        res = st.session_state['processed_data']
        st.success(T['success_msg'].format(duration=res['time']))
        
        in_b64 = base64.b64encode(res['input']).decode()
        out_b64 = base64.b64encode(res['output']).decode()
        
        st.components.v1.html(f"""
            <style>
            body {{ background: transparent; margin: 0; font-family: 'Geist', sans-serif; color: white; }}
            .player {{ background: #0a0a0a; border: 1px solid #333; border-radius: 8px; padding: 1.5rem; max-width: 600px; }}
            .ctrl {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }}
            .p-btn {{ background: white; border: none; border-radius: 50%; width: 36px; height: 36px; cursor: pointer; display: flex; align-items: center; justify-content: center; }}
            .sk {{ flex-grow: 1; accent-color: white; height: 4px; }}
            .tgl-c {{ display: flex; background: #111; border-radius: 6px; padding: 3px; border: 1px solid #333; width: fit-content; }}
            .tgl {{ padding: 6px 14px; border-radius: 4px; cursor: pointer; border: none; background: transparent; color: #888; font-size: 0.8rem; font-weight: 500; }}
            .tgl.active {{ background: #333; color: white; }}
            .time {{ font-family: monospace; font-size: 0.75rem; color: #888; }}
            </style>
            <div class="player">
                <div class="ctrl">
                    <button id="p" class="p-btn">▶</button>
                    <span id="ct" class="time">0:00</span>
                    <input type="range" id="s" class="sk" value="0" step="0.1">
                    <span id="tt" class="time">0:00</span>
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

# フッター
st.markdown("<br><br><br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(f'<div style="text-align:left;color:#888;font-size:0.85rem;padding-left:0;">{T["powered_by"]} <a href="https://github.com/Rikorose/DeepFilterNet" style="color:#fff;text-decoration:none;font-weight:600;">Hendrik Schröter (Rikorose)</a></div>', unsafe_allow_html=True)

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
        st.code("Sampling Rate: 48kHz\nModel: DeepFilterNet V3\nBackend: PyTorch / Rust")
