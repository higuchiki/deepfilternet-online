---
name: clearvoice-ai
description: ClearVoice AI（DeepFilterNet 音声ノイズ除去）の開発・デプロイ手順とルール。web_enhance.py の修正、Cloud Run デプロイ、デプロイフローの厳守、UI/UX 指針に従う際に使用する。
---

# ClearVoice AI 開発・デプロイ

## いつ使うか
- このリポジトリ（DeepFilterNet オンライン / ClearVoice AI）のコード修正・デプロイを行うとき
- `web_enhance.py`・`Dockerfile`・`requirements_cloud.txt` を触るとき
- デプロイ完了の報告をする前に確認手順を忘れないようにするとき

## デプロイフロー（厳守）

コード修正やデプロイを行う場合は、**必ず次の順で行い、5 が終わってからユーザーに完了報告すること。** プッシュだけでは報告しない。

1. **コード修正** — 要望に合わせて `web_enhance.py` や `Dockerfile` 等を修正
2. **GitHub プッシュ** — 変更を `main` にプッシュ
3. **ビルド監視** — Cloud Build のステータスを確認し、完了まで待つ（確認間隔は 1 分でよい）
4. **反映確認** — Cloud Run の公開 URL にアクセスし、修正が反映されているか確認
5. **完了報告** — 上記を確認したうえでユーザーに「反映済み」と報告

## 技術スタック

| 項目 | 内容 |
|------|------|
| アプリ | Streamlit (`web_enhance.py`) |
| インフラ | Google Cloud Run（asia-northeast1）、2 vCPU / 2GiB RAM |
| CI/CD | GitHub `main` → Cloud Build（`cloudbuild.yaml`、Kaniko ビルド・レイヤーキャッシュ） |
| Python | 3.11。PyTorch は CPU 専用（`requirements_cloud.txt`） |
| システム | Dockerfile で `ffmpeg`・`git`・`libsox-dev` 必須 |

## 依存関係

- **Python パッケージ**: `requirements_cloud.txt` で管理
- **システム**: Dockerfile の `apt-get` で `ffmpeg`・`git`・`libsox-dev`・`sox`・`libsox-fmt-all` をインストール

## UI/UX 指針

- テーマ: Next.js Docs 風のダークモード・ミニマル
- フォント: Geist / Noto Sans JP（本文は 600 程度）
- レイアウト: コンテンツは左寄せ
- 音声比較: オリジナルと処理後を、再生位置を維持したままミュート切り替えで比較できるようにする

## 参照

詳細はプロジェクトルートの [SPECIFICATION.md](../../SPECIFICATION.md) を参照する。
