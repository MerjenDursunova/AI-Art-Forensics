import os
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "0"
import streamlit as st
import torch
import clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import umap
from PIL import Image



st.set_page_config(
    page_title="AI Art Forensics",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CLF_PATH   = os.path.join(MODELS_DIR, "classifier.joblib")
UMAP_E     = os.path.join(MODELS_DIR, "umap_embeddings_sample.npy")
UMAP_M     = os.path.join(MODELS_DIR, "umap_meta_sample.csv")

COLORS = {
    "human"              : "#4C9BE8",
    "latent_diffusion"   : "#E87C4C",
    "standard_diffusion" : "#5CB85C"
}

@st.cache_resource
def load_clip():
    device = "cpu"  
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

@st.cache_resource
def load_classifier():
    return joblib.load(CLF_PATH)

@st.cache_resource
def load_umap_reference():
    """Load reference embeddings and fit UMAP reducer (cached)."""
    import os
    os.environ["NUMBA_NUM_THREADS"] = "1"  # prevent numba/MPS conflict
    
    E    = np.load(UMAP_E).astype(np.float32)
    meta = pd.read_csv(UMAP_M)
    meta["style"] = meta["style"].str.replace("-", "_")
    
    reducer = umap.UMAP(
        n_neighbors  = 30,
        min_dist     = 0.1,
        n_components = 2,
        metric       = "cosine",
        random_state = 42,
        n_jobs       = 1,      
        low_memory   = True    
    )
    coords = reducer.fit_transform(E)
    return reducer, coords, meta

def extract_embedding(img: Image.Image, model, preprocess, device) -> np.ndarray:
    """Run image through CLIP and return L2-normalised embedding."""
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor).float()
        emb = emb / (emb.norm() + 1e-8)
    return emb.cpu().numpy()

def get_saliency(img: Image.Image, model, preprocess, device) -> np.ndarray:
    """Gradient saliency: which pixels most influence the CLIP embedding."""
    model.float()
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device).float()
    tensor.requires_grad_(True)
    emb = model.encode_image(tensor).float()
    model.zero_grad()
    emb.norm().backward()
    saliency = tensor.grad.data.abs().squeeze()   
    saliency = saliency.max(dim=0)[0]             
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    if device == "cuda":
        model.half()
    return saliency.cpu().detach().numpy()

def plot_saliency(img: Image.Image, saliency: np.ndarray) -> plt.Figure:
    img_resized = img.convert("RGB").resize((224, 224))
    sal_img     = Image.fromarray(
        (saliency * 255).astype(np.uint8)
    ).resize((224, 224), Image.BILINEAR)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")

    axes[0].imshow(img_resized)
    axes[0].set_title("Original", color="white", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img_resized)
    axes[1].imshow(np.array(sal_img), cmap="jet", alpha=0.55,
                   vmin=0, vmax=255)
    axes[1].set_title("CLIP Saliency\n(red = high influence)", color="white", fontsize=11)
    axes[1].axis("off")

    for ax in axes:
        ax.set_facecolor("#0e1117")
    plt.tight_layout(pad=0.5)
    return fig

def plot_umap(coords, meta, new_coord) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for src, color in COLORS.items():
        mask = meta["source"].values == src
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, alpha=0.25, s=3, linewidths=0,
                   label=src.replace("_", " ").title())

    
    ax.scatter(new_coord[0], new_coord[1],
               c="white", s=200, marker="*", zorder=5,
               edgecolors="black", linewidths=0.8,
               label="Your image")

    ax.set_title("CLIP Embedding Space (UMAP)\n★ = your image",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    legend = ax.legend(fontsize=9, facecolor="#1e1e2e",
                       labelcolor="white", markerscale=3)
    ax.axis("off")
    plt.tight_layout()
    return fig

with st.sidebar:
    st.title("AI Art Forensics")
    st.markdown("""
**Adversarially-Aware Forensic Detection of AI-Generated Artwork**

Upload any artwork image to find out whether it was created by a human or an AI image generator.

---
**Model:** CLIP ViT-B/32 + Logistic Probe  
**Training:** AI-ArtBench (185k images)  
**Accuracy:** 99.49% (clean)  
**Styles:** 10 art styles  

---
**How it works:**
1. Your image is encoded by CLIP into a 512-dim embedding
2. A linear classifier separates Human vs AI in that space
3. A saliency map shows what pixels CLIP focused on
4. UMAP shows where your image sits in the embedding space

---
**Key findings:**
- CLIP embeddings linearly separate human vs AI art
- Surrealism & ukiyo-e are hardest to classify
- Cross-generator generalisation fails (51% OOD)
- Robust to pixel-space attacks at ε=0.02
    """)
    st.markdown("---")
    st.caption("Built by Merjen Dursunova · AI-ArtBench · CLIP ViT-B/32")


st.markdown("## Upload an artwork")
st.markdown("Supports JPG, PNG, WEBP. Works best on paintings, illustrations, and digital art.")

uploaded = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    img = Image.open(uploaded)

    
    with st.spinner("Loading CLIP model (first run takes ~30s)..."):
        model, preprocess, device = load_clip()
        clf                       = load_classifier()

    
    with st.spinner("Analysing image..."):
        emb       = extract_embedding(img, model, preprocess, device)
        p_ai      = float(clf.predict_proba(emb)[0][1])
        p_human   = 1.0 - p_ai
        label     = "AI-Generated" if p_ai >= 0.5 else "Human-Made"
        confidence = p_ai if p_ai >= 0.5 else p_human

    
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)

    with col2:
        st.markdown(f"### Verdict: **{label}**")

        color = "#E87C4C" if label == "AI-Generated" else "#4C9BE8"
        st.markdown(
            f"""
            <div style='background:{color}22; border-left:4px solid {color};
                        padding:16px; border-radius:6px; margin-bottom:16px;'>
                <span style='font-size:2rem; font-weight:700; color:{color};'>
                    {label}
                </span><br>
                <span style='font-size:1rem; color:#ccc;'>
                    Confidence: {confidence*100:.1f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence bars
        st.markdown("**Probability breakdown:**")
        st.markdown(f"Human-Made")
        st.progress(p_human, text=f"{p_human*100:.1f}%")
        st.markdown(f"AI-Generated")
        st.progress(p_ai, text=f"{p_ai*100:.1f}%")

        st.markdown("---")
        st.markdown(f"""
**What this means:**  
The detector is **{confidence*100:.1f}% confident** this image is {label.lower()}.  
{"Note: stylized genres like surrealism and ukiyo-e are harder to classify correctly." if confidence < 0.90 else ""}
        """)

    
    st.markdown("---")
    st.markdown("### Where did CLIP look?")
    st.caption(
        "The saliency map shows which pixels most strongly influenced CLIP's embedding. "
        "Red regions had the highest impact on the classification decision."
    )
    with st.spinner("Computing saliency map..."):
        try:
            saliency = get_saliency(img, model, preprocess, device)
            fig_sal  = plot_saliency(img, saliency)
            st.pyplot(fig_sal, use_container_width=True)
        except Exception as e:
            st.warning(f"Saliency map unavailable: {e}")

    
    st.markdown("---")
    st.markdown("### Where does it sit in embedding space?")
    st.caption(
        "UMAP projects 512-dimensional CLIP embeddings to 2D. "
        "The ★ shows where your image lands relative to 5,000 reference images "
        "(blue=human, orange=latent diffusion, green=standard diffusion)."
    )
    with st.spinner("Projecting into UMAP space (first run ~60s)..."):
        try:
            reducer, ref_coords, ref_meta = load_umap_reference()
            new_coord = reducer.transform(emb)[0]
            fig_umap  = plot_umap(ref_coords, ref_meta, new_coord)
            st.pyplot(fig_umap, use_container_width=True)
        except Exception as e:
            st.warning(f"UMAP projection unavailable: {e}")

    
    with st.expander("Technical details"):
        st.markdown(f"""
| Property | Value |
|---|---|
| Model | CLIP ViT-B/32 |
| Embedding dim | 512 |
| Classifier | Logistic Regression (adversarially trained) |
| P(AI) | `{p_ai:.6f}` |
| P(Human) | `{p_human:.6f}` |
| Device | `{device}` |
| Embedding norm | `{float(np.linalg.norm(emb)):.6f}` |
        """)

else:
   
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Human art** — traditional paintings, watercolors, oil on canvas, digital illustrations by human artists")
    with col2:
        st.warning("**AI art** — images generated by Stable Diffusion, Midjourney, DALL-E, or similar diffusion models")
    with col3:
        st.success("**10 styles supported** — Impressionism, Baroque, Surrealism, Ukiyo-e, Romanticism, and more")

    st.markdown("---")
    st.markdown("""
### About this project

This tool is the demo component of a forensic AI art detection system built on **CLIP ViT-B/32** embeddings.
Key findings from the research:

- **99.49% accuracy** on the AI-ArtBench benchmark (185,000 images)
- **Linear separability** — human and AI art are cleanly separated in CLIP's embedding space without task-specific fine-tuning  
- **Generator-specific signatures** — a detector trained on one generator fails (51% OOD) on another, revealing that CLIP encodes generator fingerprints, not a general AI signal
- **Stylistic ambiguity** — surrealism and ukiyo-e are hardest to classify; 7/10 confused images are human art near AI clusters
- **Pixel-space robustness** — CLIP's encoder resists imperceptible pixel perturbations (ε=0.02), though the linear classifier is fragile in embedding space
    """)
