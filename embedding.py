# app.py
# -*- coding: utf-8 -*-
"""
国試過去問 Excel（テキスト＋画像URL）を読み込み、
CLIPでテキスト＆画像を埋め込み → 融合ベクトルを作成。
結果をCSVでダウンロードしつつ、t-SNEで2Dマッピングして可視化する Streamlit アプリ。
"""

import io
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.manifold import TSNE

import streamlit as st
import torch
import clip  # openai/CLIP
import altair as alt


# ------------------------------------------------------------
# CLIP モデルのロード（キャッシュする）
# ------------------------------------------------------------
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def embed_text_clip(model, device, text: str) -> np.ndarray:
    """テキストを CLIP で 512次元ベクトルに埋め込む"""
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(512, dtype=np.float32)

    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0].astype(np.float32)


def drive_link_to_direct_url(link: str) -> Optional[str]:
    """Google Drive 共有リンクを ダウンロードURL に変換"""
    if not isinstance(link, str):
        return None
    m = re.search(r"/d/([^/]+)/", link)
    if not m:
        return None
    file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def load_image_from_drive(link: str) -> Optional[Image.Image]:
    """Google Drive リンクから画像を取得して PIL に変換"""
    url = drive_link_to_direct_url(link)
    if url is None:
        return None
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception:
        return None


def embed_image_clip(model, preprocess, device, link: str) -> np.ndarray:
    """画像を512次元のCLIP埋め込みに"""
    if not isinstance(link, str) or link.strip() == "":
        return np.zeros(512, dtype=np.float32)

    img = load_image_from_drive(link)
    if img is None:
        return np.zeros(512, dtype=np.float32)

    img_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0].astype(np.float32)


def fuse_embeddings(text_vec: np.ndarray, image_vec: np.ndarray,
                    alpha: float = 0.7, beta: float = 0.3) -> np.ndarray:
    """テキストと画像のベクトルを重み付きで融合"""
    v = alpha * text_vec + beta * image_vec
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


# ============================================================
# Streamlit アプリ UI
# ============================================================
st.set_page_config(page_title="国試DB 埋め込み＋t-SNE可視化", layout="wide", page_icon="🦷")

st.title("🦷 国試過去問：テキスト＋画像 埋め込み生成 & t-SNE 可視化アプリ")

uploaded = st.file_uploader("国試DB Excel (.xlsx) をアップロード", type=["xlsx"])

if uploaded is not None:
    df = pd.read_excel(uploaded)
    st.success(f"読み込み成功：{df.shape[0]} 行 × {df.shape[1]} 列")

    cols = list(df.columns)

    # ID列推定
    default_id_col = None
    for cand in ["問題ID", "id", "ID"]:
        if cand in cols:
            default_id_col = cand
            break

    # 画像列推定
    default_img_col = None
    for cand in ["画像URL", "img_url", "image_url"]:
        if cand in cols:
            default_img_col = cand
            break

    st.subheader("列の設定")
    id_col = st.selectbox("問題ID", options=cols,
                          index=cols.index(default_id_col) if default_id_col else 0)

    default_text_cols = [c for c in ["問題文", "a", "b", "c", "d", "e", "解説"] if c in cols]
    text_cols = st.multiselect("テキストに使う列", options=cols,
                               default=default_text_cols if default_text_cols else [cols[0]])

    img_col = st.selectbox("画像URL列（Driveリンク）",
                           options=["（なし）"] + cols,
                           index=(cols.index(default_img_col) + 1) if default_img_col else 0)

    use_image = img_col != "（なし）"

    st.subheader("埋め込みの重み")
    alpha = st.slider("テキスト重み α", 0.0, 1.0, 0.7, 0.05)
    beta = 1.0 - alpha

    st.subheader("t-SNE 設定")
    perplexity = st.slider("perplexity", 5, 50, 30, 1)
    tsne_seed = st.number_input("random_state", value=42)

    color_col = st.selectbox("プロットの色分け（任意）",
                             options=["（なし）"] + cols)

    run_button = st.button("埋め込み＋t-SNE 実行", type="primary")

    if run_button:
        model, preprocess, device = load_clip_model()

        fused_vecs = []
        progress = st.progress(0)
        n = len(df)

        for i, row in df.iterrows():
            # テキスト結合
            texts = []
            for c in text_cols:
                v = row.get(c, "")
                if isinstance(v, str):
                    texts.append(v.strip())
            full_text = "\n".join(texts)

            text_vec = embed_text_clip(model, device, full_text)

            if use_image:
                image_vec = embed_image_clip(model, preprocess, device, row.get(img_col, ""))
            else:
                image_vec = np.zeros(512, dtype=np.float32)

            vec = fuse_embeddings(text_vec, image_vec, alpha=alpha, beta=beta)
            fused_vecs.append(vec)

            if (i + 1) % 10 == 0:
                progress.progress((i + 1) / n)

        progress.progress(1.0)

        fused_arr = np.vstack(fused_vecs)
        emb_df = pd.DataFrame({"問題ID": df[id_col]})
        for i in range(fused_arr.shape[1]):
            emb_df[f"emb_{i}"] = fused_arr[:, i]

        st.download_button(
            "⬇ 埋め込みCSVをダウンロード",
            emb_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="kokushi_embedding_fused.csv",
            mime="text/csv"
        )

        st.subheader("t-SNE 実行中…")
        tsne = TSNE(
            n_components=2,
            perplexity=int(perplexity),
            random_state=int(tsne_seed),
            metric="cosine",
            init="random",
        )
        coords = tsne.fit_transform(fused_arr)

        vis_df = df.copy()
        vis_df["tsne_x"] = coords[:, 0]
        vis_df["tsne_y"] = coords[:, 1]

        chart = (
            alt.Chart(vis_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x="tsne_x:Q",
                y="tsne_y:Q",
                tooltip=[id_col] + text_cols[:2],
                color=(alt.Color(f"{color_col}:N") if color_col != "（なし）" else alt.value("steelblue")),
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

        st.download_button(
            "⬇ t-SNE座標CSV",
            vis_df[[id_col, "tsne_x", "tsne_y"] + ([color_col] if color_col != "（なし）" else [])]
            .to_csv(index=False).encode("utf-8-sig"),
            file_name="kokushi_tsne.csv",
            mime="text/csv"
        )


else:
    st.info("Excelファイルをアップロードしてください。")
