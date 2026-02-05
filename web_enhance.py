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
    'version': 'v0.1.0-beta',
    'subtitle': 'AIノイズ処理ライブラリDeepFilterNetを使ったノイズ処理Webアプリ',
    'step1': '1. 音源をアップロード',
    'uploader_label': 'WAV, M4A, MP3, AAC ファイルを選択してください',
    'step2': '2. 除去強度の設定',
    'step2_hint': '※わからなければ初期設定のままで良いです',
    'atten_label': 'ノイズ除去の制限 (dB)',
    'atten_help': '0dBに近いほど強力にノイズを消します。声が不自然な場合のみ値を大きくしてください。',
    'btn_enhance': 'Process Audio',
    'status_preparing': '音声を準備中...',
    'status_processing': 'AIがノイズを解析・除去しています...',
    'status_saving': '結果を生成中...',
    'status_done': 'Done! {duration:.1f}s',
    'step3': '3. 処理結果',
    'success_msg': 'Success  \n{duration:.1f}s',
    'input_label': '元の音源',
    'output_label': 'AI除去後',
    'btn_download': 'Download',
    'dl_wav': 'WAV',
    'dl_mp3': 'MP3',
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
        font-size: 1.5rem !important; /* サイズを大幅に抑えてモダンに */
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #ffffff;
        text-align: left;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .version-badge {
        font-size: 0.65rem;
        background: #1a1a1a;
        color: #888;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #333;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .sub-title {
        color: var(--muted);
        font-size: 0.9rem; /* さらに小さく */
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
        font-size: 1rem !important; /* ラベルのような控えめなサイズ */
        letter-spacing: 0.02em !important;
        text-align: left !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
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
    
    /* 共通ボタンスタイル (Vercelスタイル) */
    .stButton > button, 
    .stDownloadButton > button, 
    button[data-testid="stBaseButton-secondary"]:not([aria-label="Remove file"]),
    div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        height: 2.8rem !important;
        width: auto !important;
        min-width: 160px;
        padding: 0 2rem !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid #ffffff !important;
        margin-top: 1rem !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-family: 'Geist', 'Noto Sans JP', sans-serif !important;
    }
    /* ボタン内部のテキスト要素（p, span等）に対しても強制的にセミボールドを適用 */
    .stButton > button *, 
    .stDownloadButton > button *, 
    button[data-testid="stBaseButton-secondary"]:not([aria-label="Remove file"]) *,
    div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] * {
        font-weight: 600 !important;
        color: inherit !important;
        font-family: inherit !important;
    }
    .stButton > button:hover, 
    .stDownloadButton > button:hover, 
    button[data-testid="stBaseButton-secondary"]:not([aria-label="Remove file"]):hover,
    div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        transform: translateY(-1px);
    }
    .stButton > button:active, 
    .stDownloadButton > button:active, 
    button[data-testid="stBaseButton-secondary"]:not([aria-label="Remove file"]):active,
    div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]:active {
        transform: translateY(0);
        opacity: 0.8;
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
    
    /* 成功メッセージのカスタマイズ */
    .success-box {
        padding: 1.25rem;
        background: #0a0a0a;
        border-radius: 8px;
        border: 1px solid #333333;
        margin-bottom: 1.5rem;
        text-align: left;
        max-width: fit-content;
        min-width: 120px;
    }
    .success-box .status {
        font-weight: 600;
        font-size: 0.85rem;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .success-box .time {
        font-family: 'Geist Mono', monospace;
        font-size: 1.1rem;
        color: #ffffff;
        font-weight: 500;
    }

    /* Xリンクのスタイル */
    .x-link {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999999;
        text-decoration: none !important;
        opacity: 0.5;
        transition: opacity 0.2s ease;
        display: flex;
        align-items: center;
        gap: 8px;
        background: transparent !important;
    }
    .x-link:hover {
        opacity: 1;
    }
    .x-link .x-text {
        font-size: 0.7rem;
        color: #888;
        white-space: nowrap;
    }
    .x-link:hover .x-text {
        color: #ffffff;
    }

    /* エクスパンダー（折りたたみ）のカスタマイズ */
    .stExpander {
        border: none !important;
        background: transparent !important;
        max-width: fit-content !important;
    }
    .stExpander details {
        border: none !important;
    }
    .stExpander summary {
        color: var(--muted) !important;
        font-size: 0.85rem !important;
        padding: 0 !important;
        transition: color 0.2s ease;
    }
    .stExpander summary:hover {
        color: #ffffff !important;
    }
    .stExpander summary svg {
        display: none !important; /* 矢印を消してさらにミニマルに */
    }

    .beta-notice {
        color: var(--muted);
        font-size: 0.75rem;
        margin-top: 1rem;
        text-align: left;
    }

    /* Streamlit要素の非表示 */
    #MainMenu, footer, header, div[data-testid="stDecoration"], div[data-testid="stHeader"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Xアイコン（右下に固定）
st.markdown(f"""
    <a href="https://x.com/HiguchiKi" target="_blank" class="x-link">
        <span class="x-text">エラー報告・レビューはこちら</span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
            <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
        </svg>
    </a>
""", unsafe_allow_html=True)

# メインコンテンツ
st.markdown(f'<h1 class="main-title">{T["title"]} <span class="version-badge">{T["version"]}</span></h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title" style="margin-bottom: 3rem;">{T["subtitle"]}</p>', unsafe_allow_html=True)

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
    st.markdown(f'<p style="color: var(--muted); font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1rem;">{T["step2_hint"]}</p>', unsafe_allow_html=True)
    col_conf1, col_conf2 = st.columns([2, 1])
    with col_conf1:
        atten_lim = st.slider(T['atten_label'], 0, 100, 0, help=T['atten_help'])
        
        if st.button(T['btn_enhance']):
            if 'processed_data' in st.session_state:
                del st.session_state['processed_data']
                
            with st.status(T['status_processing'], expanded=True) as status:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    try:
                        input_path = os.path.join(tmpdirname, uploaded_file.name)
                        with open(input_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        st.write(T['status_preparing'])
                        load_path = input_path
                        if not input_path.lower().endswith(".wav"):
                            temp_wav = os.path.join(tmpdirname, "temp.wav")
                            # ffmpegの出力を詳細に取得
                            result = subprocess.run(["ffmpeg", "-y", "-i", input_path, temp_wav], capture_output=True, text=True)
                            if result.returncode != 0:
                                st.error(f"FFmpeg Error: {result.stderr}")
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
                        # MP3 を ffmpeg で生成（Download の形式選択用）
                        mp3_path = os.path.join(tmpdirname, "enhanced.mp3")
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", output_path, "-acodec", "libmp3lame", "-q:a", "2", mp3_path],
                            capture_output=True, timeout=120
                        )
                        output_mp3 = b""
                        if os.path.isfile(mp3_path):
                            with open(mp3_path, "rb") as f:
                                output_mp3 = f.read()
                        # プレイヤー用に元音源もWAVで保存（シーク同期のため）
                        input_wav_path = os.path.join(tmpdirname, "original.wav")
                        save_audio(input_wav_path, audio, sr=df_state.sr())
                        with open(input_wav_path, "rb") as f:
                            input_wav_bytes = f.read()
                        
                        st.session_state['processed_data'] = {
                            'input_wav': input_wav_bytes,
                            'output': audio_bytes,
                            'output_mp3': output_mp3,
                            'name': uploaded_file.name,
                            'time': proc_duration
                        }
                        status.update(label=T['status_done'].format(duration=proc_duration), state="complete")
                        
                        # Success表示直後にプレイヤーが出るまでの間に空のプレースホルダーでローディングを維持
                        with st.spinner("結果を表示しています..."):
                            time.sleep(0.5) # 描画の安定化のためのわずかな待ち時間
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        status.update(label="❌ Error", state="error")

    if 'processed_data' in st.session_state:
        res = st.session_state['processed_data']
        in_b64 = base64.b64encode(res['input_wav']).decode()
        out_b64 = base64.b64encode(res['output']).decode()
        output_mp3 = res.get('output_mp3') or b""
        mp3_b64 = base64.b64encode(output_mp3).decode() if output_mp3 else ""
        has_mp3 = "true" if output_mp3 else "false"
        base_name = os.path.splitext(res['name'])[0]
        dl_name_wav = base_name + "_enhanced.wav"
        dl_name_mp3 = base_name + "_enhanced.mp3"
        dl_name_wav_esc = dl_name_wav.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        dl_name_mp3_esc = dl_name_mp3.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        
        st.subheader(T['step3'])
        
        # 成功メッセージ
        st.markdown(f"""
            <div class="success-box">
                <div class="status">Success</div>
                <div class="time">{res['time']:.1f}s</div>
            </div>
        """, unsafe_allow_html=True)
        
        # プレイヤー: Blob URL で再生を軽く / UI 統一 / WAV・MP3 ダウンロード
        st.components.v1.html(f"""
            <style>
                .player-wrap {{ max-width: 560px; margin: 1rem 0; font-family: inherit; }}
                .player-src {{ display: flex; gap: 8px; margin-bottom: 14px; }}
                .player-src button {{
                    padding: 6px 14px; border-radius: 6px; font-size: 0.8rem; font-weight: 500;
                    background: #1a1a1a; color: #e5e5e5; border: 1px solid #333; cursor: pointer;
                }}
                .player-src button.active {{ background: #333; color: #fff; border-color: #555; }}
                .player-src button:hover {{ background: #262626; }}
                .player-ctrl {{ display: flex; align-items: center; gap: 6px; margin-bottom: 10px; }}
                .player-ctrl button {{
                    width: 36px; height: 36px; border-radius: 8px; border: 1px solid #333;
                    background: #1a1a1a; color: #e5e5e5; cursor: pointer; font-size: 0.9rem;
                    display: flex; align-items: center; justify-content: center; padding: 0;
                }}
                .player-ctrl button:hover {{ background: #262626; border-color: #444; }}
                .player-ctrl .skip {{ width: auto; padding: 0 10px; font-size: 0.75rem; }}
                .player-time {{ color: #888; font-size: 0.8rem; margin-bottom: 6px; font-variant-numeric: tabular-nums; }}
                .player-seek {{ width: 100%; height: 6px; border-radius: 3px; accent-color: #fff; cursor: pointer; margin-bottom: 16px; }}
                .player-dl {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
                .player-dl select {{
                    padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;
                    background: #1a1a1a; color: #e5e5e5; border: 1px solid #333; cursor: pointer;
                }}
                .player-dl .dl-btn {{
                    padding: 10px 20px; border-radius: 6px; font-size: 0.9rem; font-weight: 600;
                    background: #fff; color: #000; border: 1px solid #fff; cursor: pointer;
                }}
                .player-dl .dl-btn:hover {{ background: #e5e5e5; border-color: #e5e5e5; }}
            </style>
            <div class="player-wrap">
                <div class="player-src">
                    <button type="button" id="btnOrig">{T['input_label']}</button>
                    <button type="button" id="btnEnh">{T['output_label']}</button>
                </div>
                <div class="player-ctrl">
                    <span id="loadStatus" style="color:#888;font-size:0.8rem;margin-right:8px;"></span>
                    <button type="button" id="btnPlay" title="再生">▶</button>
                    <button type="button" id="btnPause" title="一時停止">⏸</button>
                    <button type="button" id="btnStop" title="停止">⏹</button>
                    <button type="button" id="btnBack10" class="skip" title="10秒戻る">−10</button>
                    <button type="button" id="btnFwd10" class="skip" title="10秒進む">+10</button>
                </div>
                <div class="player-time" id="timeDisplay">0:00 / 0:00</div>
                <input type="range" class="player-seek" id="seekBar" min="0" max="100" value="0" step="0.1">
                <div class="player-dl">
                    <select id="dlFormat">
                        <option value="wav">{T['dl_wav']}</option>
                        <option value="mp3" id="optMp3">{T['dl_mp3']}</option>
                    </select>
                    <button type="button" id="btnDownload" class="dl-btn">{T['btn_download']}</button>
                </div>
            </div>
            <textarea id="storeIn" style="display:none;width:0;height:0;">{in_b64}</textarea>
            <textarea id="storeOut" style="display:none;width:0;height:0;">{out_b64}</textarea>
            <audio id="a1" preload="auto"></audio>
            <audio id="a2" preload="auto"></audio>
            <script>
                (function() {{
                    var a1 = document.getElementById('a1');
                    var a2 = document.getElementById('a2');
                    var seekBar = document.getElementById('seekBar');
                    var timeDisplay = document.getElementById('timeDisplay');
                    var btnOrig = document.getElementById('btnOrig');
                    var btnEnh = document.getElementById('btnEnh');
                    var btnPlay = document.getElementById('btnPlay');
                    var btnPause = document.getElementById('btnPause');
                    var btnStop = document.getElementById('btnStop');
                    var btnBack10 = document.getElementById('btnBack10');
                    var btnFwd10 = document.getElementById('btnFwd10');
                    var btnDownload = document.getElementById('btnDownload');
                    var dlFormat = document.getElementById('dlFormat');
                    var optMp3 = document.getElementById('optMp3');
                    var active = 1;
                    var dur = 0;
                    var hasMp3 = {has_mp3};
                    var dlNameWav = '{dl_name_wav_esc}';
                    var dlNameMp3 = '{dl_name_mp3_esc}';
                    var mp3B64 = '{mp3_b64}';
                    var blob1, blob2;
                    var loadStatus = document.getElementById('loadStatus');
                    function b64ToBlob(b64, type) {{
                        var bin = atob(b64);
                        var buf = new Uint8Array(bin.length);
                        for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
                        return new Blob([buf], {{ type: type }});
                    }}
                    function initAudio() {{
                        loadStatus.textContent = 'Preparing…';
                        btnPlay.disabled = true;
                        var inB64 = document.getElementById('storeIn').value;
                        var outB64 = document.getElementById('storeOut').value;
                        blob1 = b64ToBlob(inB64, 'audio/wav');
                        blob2 = b64ToBlob(outB64, 'audio/wav');
                        a1.src = URL.createObjectURL(blob1);
                        a2.src = URL.createObjectURL(blob2);
                        a1.preload = 'auto';
                        a2.preload = 'auto';
                        a1.load();
                        a2.load();
                        var ready = 0;
                        function onReady() {{
                            ready++;
                            if (ready >= 2) {{
                                loadStatus.textContent = '';
                                btnPlay.disabled = false;
                            }}
                        }}
                        a1.addEventListener('loadeddata', onReady, {{ once: true }});
                        a2.addEventListener('loadeddata', onReady, {{ once: true }});
                    }}
                    if (typeof requestIdleCallback !== 'undefined')
                        requestIdleCallback(initAudio, {{ timeout: 400 }});
                    else
                        setTimeout(initAudio, 0);
                    if (!hasMp3) {{ optMp3.disabled = true; optMp3.textContent = optMp3.textContent + ' (n/a)'; }}
                    function curr() {{ return active === 1 ? a1 : a2; }}
                    function fmt(t) {{
                        if (isNaN(t) || !isFinite(t)) return '0:00';
                        var m = Math.floor(t / 60), s = Math.floor(t % 60);
                        return m + ':' + (s < 10 ? '0' : '') + s;
                    }}
                    function setActive(n) {{
                        active = n;
                        btnOrig.classList.toggle('active', n === 1);
                        btnEnh.classList.toggle('active', n === 2);
                        a1.muted = (n !== 1);
                        a2.muted = (n !== 2);
                        if (n === 1) {{ a2.pause(); a2.currentTime = a1.currentTime; a1.play(); }}
                        else {{ a1.pause(); a1.currentTime = a2.currentTime; a2.play(); }}
                    }}
                    btnOrig.onclick = function() {{ setActive(1); }};
                    btnEnh.onclick = function() {{ setActive(2); }};
                    btnPlay.onclick = function() {{ curr().play(); }};
                    btnPause.onclick = function() {{ a1.pause(); a2.pause(); }};
                    btnStop.onclick = function() {{
                        a1.pause(); a2.pause();
                        a1.currentTime = a2.currentTime = 0;
                        seekBar.value = 0;
                        timeDisplay.textContent = '0:00 / ' + fmt(dur);
                    }};
                    btnBack10.onclick = function() {{
                        var t = Math.max(0, curr().currentTime - 10);
                        a1.currentTime = a2.currentTime = t;
                        seekBar.value = t;
                        timeDisplay.textContent = fmt(t) + ' / ' + fmt(dur);
                    }};
                    btnFwd10.onclick = function() {{
                        var t = Math.min(dur, curr().currentTime + 10);
                        a1.currentTime = a2.currentTime = t;
                        seekBar.value = t;
                        timeDisplay.textContent = fmt(t) + ' / ' + fmt(dur);
                    }};
                    btnDownload.onclick = function() {{
                        try {{
                            var blob, name, mime;
                            if (dlFormat.value === 'mp3' && hasMp3 && mp3B64) {{
                                blob = b64ToBlob(mp3B64, 'audio/mpeg');
                                name = dlNameMp3;
                            }} else {{
                                blob = blob2;
                                name = dlNameWav;
                            }}
                            var url = URL.createObjectURL(blob);
                            var a = document.createElement('a');
                            a.href = url;
                            a.download = name;
                            a.click();
                            URL.revokeObjectURL(url);
                        }} catch (e) {{ console.error(e); }}
                    }};
                    a1.onloadedmetadata = a2.onloadedmetadata = function() {{
                        dur = Math.max(a1.duration || 0, a2.duration || 0);
                        seekBar.max = dur;
                    }};
                    seekBar.oninput = function() {{
                        var t = parseFloat(seekBar.value);
                        a1.currentTime = a2.currentTime = t;
                        timeDisplay.textContent = fmt(t) + ' / ' + fmt(dur);
                    }};
                    function onTime() {{
                        var t = active === 1 ? a1.currentTime : a2.currentTime;
                        a1.currentTime = a2.currentTime = t;
                        seekBar.value = t;
                        timeDisplay.textContent = fmt(t) + ' / ' + fmt(dur);
                    }}
                    a1.ontimeupdate = a2.ontimeupdate = onTime;
                    a1.onloadedmetadata();
                }})();
            </script>
        """, height=240)

# フッター
st.markdown("<br><br><br><br>", unsafe_allow_html=True)
st.divider()
st.markdown('<div class="beta-notice">※現在開発中のベータ版です。予期せぬ動作が発生する可能性があります。</div>', unsafe_allow_html=True)
st.markdown(f'<div style="text-align:left;color:#888;font-size:0.85rem;padding-left:0;margin-top:0.5rem;">{T["powered_by"]} <a href="https://github.com/Rikorose/DeepFilterNet" style="color:#fff;text-decoration:none;font-weight:600;">Hendrik Schröter (Rikorose)</a></div>', unsafe_allow_html=True)

with st.expander("ドキュメント・技術仕様を表示"):
    st.markdown(f"""
        <div style="text-align:left; color:#888; font-size:0.85rem; margin-bottom:1.5rem; line-height:1.6;">
            {T["powered_by"]} <a href="https://github.com/Rikorose/DeepFilterNet" style="color:#fff; text-decoration:none; font-weight:600;">Hendrik Schröter (Rikorose)</a><br>
            Developed by <a href="https://x.com/HiguchiKi" target="_blank" style="color:#fff; text-decoration:none; font-weight:600;">@HiguchiKi</a>
        </div>
    """, unsafe_allow_html=True)
    
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.markdown("### 概要")
        st.markdown("""
        **ディープフィルタリング**
        AIによる周波数分離技術。従来のゲート処理とは異なり、背景ノイズのみを除去し、声の質感を高いクオリティで維持します。

        **パフォーマンス**
        Rustで書かれた高速エンジンにより、一般的なCPU環境でもリアルタイムに近い速度で処理が可能です。

        **プライバシー**
        アップロードされたファイルはメモリ上でのみ処理され、セッション終了後に自動的に破棄されます。サーバーに保存されることはありません。
        """)
    with exp_col2:
        st.markdown("### 技術仕様")
        st.code("サンプリングレート: 48kHz\nモデル: DeepFilterNet V3\nバックエンド: PyTorch / Rust")
