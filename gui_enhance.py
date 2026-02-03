import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import torch
import numpy as np
import sounddevice as sd
import time
from df.enhance import enhance, init_df, load_audio, save_audio

class DeepFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepFilterNet Audio Enhancer")
        self.root.geometry("500x650")

        self.input_path = tk.StringVar()
        self.attenuation = tk.DoubleVar(value=0)
        self.post_filter = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="準備完了")
        
        # 再生用の状態
        self.original_audio_np = None
        self.enhanced_audio_np = None
        self.current_sr = None
        self._is_playing = False
        self.play_source = tk.StringVar(value="enhanced") # "original" or "enhanced"
        self.stream = None
        self.play_ptr = 0

        self.create_widgets()
        
        # モデルの初期化（バックグラウンドで行う）
        self.model = None
        self.df_state = None
        threading.Thread(target=self.initialize_model, daemon=True).start()

    def create_widgets(self):
        padding = {"padx": 20, "pady": 10}

        # ファイル選択
        file_frame = ttk.LabelFrame(self.root, text="音声ファイル選択")
        file_frame.pack(fill="x", **padding)
        
        ttk.Entry(file_frame, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=(5, 5), pady=5)
        ttk.Button(file_frame, text="参照", command=self.browse_file).pack(side="right", padx=5, pady=5)

        # パラメータ設定
        param_frame = ttk.LabelFrame(self.root, text="パラメータ設定")
        param_frame.pack(fill="x", **padding)

        # Attenuation スライダー
        ttk.Label(param_frame, text="ノイズ減衰の制限 (dB):").pack(anchor="w", padx=5)
        ttk.Label(param_frame, text="※0で制限なし（最大除去）、値を大きくするとノイズを残します", font=("", 10)).pack(anchor="w", padx=5)
        scale = ttk.Scale(param_frame, from_=0, to=100, variable=self.attenuation, orient="horizontal")
        scale.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(param_frame, textvariable=self.attenuation).pack(anchor="e", padx=5)

        # 実行ボタン
        self.run_button = ttk.Button(self.root, text="ノイズ除去を開始", command=self.start_enhancement, state="disabled")
        self.run_button.pack(pady=10)

        # 進捗表示フレーム
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill="x", padx=20)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", side="top", pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(side="top")

        # プレビュー再生・比較フレーム
        preview_frame = ttk.LabelFrame(self.root, text="プレビュー・比較再生")
        preview_frame.pack(fill="x", **padding)
        
        # タイムラインスライダー
        self.timeline_var = tk.DoubleVar(value=0)
        self.is_dragging = False
        self.timeline_scale = ttk.Scale(
            preview_frame, from_=0, to=100, variable=self.timeline_var, 
            orient="horizontal", command=self.on_timeline_click
        )
        self.timeline_scale.pack(fill="x", padx=5, pady=5)
        self.timeline_scale.bind("<ButtonRelease-1>", self.on_timeline_release)
        self.timeline_scale.bind("<Button-1>", self.on_timeline_press)

        # 時間表示
        self.time_label = ttk.Label(preview_frame, text="00:00 / 00:00")
        self.time_label.pack()

        self.play_button = ttk.Button(preview_frame, text="再生", command=self.toggle_playback, state="disabled")
        self.play_button.pack(pady=5)

        # ソース切り替えボタン
        switch_frame = ttk.Frame(preview_frame)
        switch_frame.pack(pady=5)
        
        self.orig_radio = ttk.Radiobutton(switch_frame, text="元の音源", variable=self.play_source, value="original", state="disabled")
        self.orig_radio.pack(side="left", padx=10)
        
        self.enh_radio = ttk.Radiobutton(switch_frame, text="除去後", variable=self.play_source, value="enhanced", state="disabled")
        self.enh_radio.pack(side="left", padx=10)

        # ステータス表示
        status_label = ttk.Label(self.root, textvariable=self.status_text, foreground="blue")
        status_label.pack(pady=10)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.m4a *.mp3 *.flac")])
        if filename:
            self.input_path.set(filename)

    def initialize_model(self):
        self.status_text.set("モデルを初期化中...")
        try:
            self.model, self.df_state, _ = init_df()
            self.status_text.set("準備完了")
            self.run_button.config(state="normal")
        except Exception as e:
            self.status_text.set(f"初期化エラー: {str(e)}")

    def start_enhancement(self):
        input_file = self.input_path.get()
        if not input_file:
            messagebox.showwarning("警告", "ファイルを選択してください")
            return
        
        self.stop_playback()
        self.run_button.config(state="disabled")
        self.play_button.config(state="disabled")
        self.orig_radio.config(state="disabled")
        self.enh_radio.config(state="disabled")
        self.status_text.set("処理中...")
        self.progress_var.set(0)
        self.progress_label.config(text="開始しています...")
        
        # 処理を別スレッドで実行
        threading.Thread(target=self.process_audio, args=(input_file,), daemon=True).start()

    def process_audio(self, input_path):
        start_time = time.time()
        try:
            input_path = os.path.abspath(input_path)
            base, ext = os.path.splitext(input_path)
            output_path = base + "_enhanced.wav"

            self.status_text.set("読み込み中...")
            self.progress_var.set(5)
            self.progress_label.config(text="ファイルを準備しています...")
            
            temp_wav = None
            if ext.lower() in [".m4a", ".mp3", ".mp4", ".aac"]:
                self.status_text.set("フォーマット変換中...")
                self.progress_var.set(10)
                temp_wav = base + "_temp_conv.wav"
                import subprocess
                cmd = ["ffmpeg", "-y", "-i", input_path, temp_wav]
                subprocess.run(cmd, check=True, capture_output=True)
                load_path = temp_wav
            else:
                load_path = input_path

            audio, _ = load_audio(load_path, sr=self.df_state.sr())
            
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

            self.status_text.set("ノイズ除去中...")
            self.progress_var.set(20)
            self.progress_label.config(text="AIがノイズを解析・除去しています...")

            atten_lim = self.attenuation.get()
            proc_start = time.time()
            
            enhanced = enhance(
                self.model, 
                self.df_state, 
                audio, 
                atten_lim_db=atten_lim
            )
            
            proc_end = time.time()
            duration = proc_end - proc_start
            
            self.progress_var.set(90)
            self.progress_label.config(text=f"保存中... (処理時間: {duration:.1f}秒)")

            self.original_audio_np = audio.t().cpu().numpy()
            self.enhanced_audio_np = enhanced.t().cpu().numpy()
            self.current_sr = self.df_state.sr()
            
            total_frames = len(self.enhanced_audio_np)
            self.timeline_scale.config(to=total_frames)
            self.update_time_label(0, total_frames)
            
            save_audio(output_path, enhanced, sr=self.df_state.sr())
            
            self.progress_var.set(100)
            total_elapsed = time.time() - start_time
            self.status_text.set("完了！")
            self.progress_label.config(text=f"完了！ 総処理時間: {total_elapsed:.1f}秒")
            
            self.play_button.config(state="normal")
            self.orig_radio.config(state="normal")
            self.enh_radio.config(state="normal")
            messagebox.showinfo("成功", f"ノイズ除去が完了しました：\n{output_path}")
        except Exception as e:
            self.status_text.set("エラー発生")
            self.progress_label.config(text="エラーにより中断しました")
            print(f"Error detail: {str(e)}")
            messagebox.showerror("エラー", f"処理中にエラーが発生しました：\n{str(e)}")
        finally:
            self.run_button.config(state="normal")

    def update_time_label(self, current_frame, total_frames):
        curr_sec = int(current_frame / self.current_sr)
        total_sec = int(total_frames / self.current_sr)
        self.time_label.config(text=f"{curr_sec//60:02}:{curr_sec%60:02} / {total_sec//60:02}:{total_sec%60:02}")

    def on_timeline_press(self, event):
        self.is_dragging = True

    def on_timeline_release(self, event):
        self.is_dragging = False
        if self.enhanced_audio_np is not None:
            self.play_ptr = int(self.timeline_var.get())

    def on_timeline_click(self, val):
        if self.is_dragging and self.enhanced_audio_np is not None:
            self.update_time_label(int(float(val)), len(self.enhanced_audio_np))

    def toggle_playback(self):
        if self._is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.enhanced_audio_np is None:
            return
        
        self._is_playing = True
        self.play_button.config(text="停止")
        self.play_ptr = int(self.timeline_var.get())
        
        # GUI更新を制限するためのタイマー
        self.last_gui_update = 0
        
        # キャッシュしてアクセスを高速化
        orig_audio = self.original_audio_np
        enh_audio = self.enhanced_audio_np
        total_len = len(enh_audio)

        def callback(outdata, frames, time_info, status):
            if not self._is_playing:
                raise sd.CallbackStop()
            
            ptr = self.play_ptr
            chunk_size = min(frames, total_len - ptr)
            if chunk_size <= 0:
                raise sd.CallbackStop()
            
            # 選択されているソースに応じてデータをコピー
            if self.play_source.get() == "original":
                outdata[:chunk_size] = orig_audio[ptr:ptr+chunk_size]
            else:
                outdata[:chunk_size] = enh_audio[ptr:ptr+chunk_size]
            
            if chunk_size < frames:
                outdata[chunk_size:] = 0
                self.play_ptr = total_len
                raise sd.CallbackStop()
            
            self.play_ptr += chunk_size
            
            # GUIの更新頻度を落とす (200msごとにしてさらに負荷軽減)
            now = time.time()
            if now - self.last_gui_update > 0.2:
                if not self.is_dragging:
                    # afterはメインスレッドで実行されるため、頻度が高いと重くなる
                    self.root.after_idle(self._update_gui_from_playback, self.play_ptr)
                self.last_gui_update = now

        try:
            # blocksizeをさらに大きくして安定性を向上
            self.stream = sd.OutputStream(
                samplerate=self.current_sr,
                channels=enh_audio.shape[1],
                callback=callback,
                finished_callback=self.stop_playback,
                blocksize=4096
            )
            self.stream.start()
        except Exception as e:
            messagebox.showerror("エラー", f"再生エラー: {str(e)}")
            self.stop_playback()

    def _update_gui_from_playback(self, ptr):
        if self._is_playing and not self.is_dragging:
            self.timeline_var.set(ptr)
            self.update_time_label(ptr, len(self.enhanced_audio_np))

    def stop_playback(self):
        self._is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.play_button.config(text="再生")

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepFilterGUI(root)
    root.mainloop()
