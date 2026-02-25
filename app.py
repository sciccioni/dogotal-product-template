import io
import json
import re
import base64
import urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# ---------------------------------------------------------------
# 1. GPT-4o mini Vision — localizza la ZONA dell'anno
# ---------------------------------------------------------------
def trova_anno_con_gpt(img_pil: Image.Image, openai_api_key: str) -> dict | None:
    buf = io.BytesIO()
    img_pil.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "Analizza questa copertina di fotolibro. "
        "Trova SOLO il testo che rappresenta un anno numerico (formato 20XX, es. 2024, 2025, 2026). "
        "NON includere il titolo principale (nome città o luogo).\n\n"
        "Rispondi ESCLUSIVAMENTE con JSON valido, senza markdown:\n"
        '{"trovato": true, "testo": "2025", "x_pct": 57.0, "y_pct": 25.0, "w_pct": 32.0, "h_pct": 8.5}\n\n'
        "x_pct/y_pct = angolo top-left in %, w_pct/h_pct = dimensioni in %.\n"
        'Se non trovi nessun anno: {"trovato": false}'
    )

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "max_tokens": 150,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"
                }},
                {"type": "text", "text": prompt}
            ]
        }]
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            raw = data["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            return result if result.get("trovato") else None
    except Exception as e:
        st.error(f"❌ Errore GPT-4o mini: {e}")
        return None

# ---------------------------------------------------------------
# 2. Raffina bbox con scansione pixel reale
# ---------------------------------------------------------------
def raffina_bbox_pixel(img_arr, gpt_x1, gpt_y1, gpt_x2, gpt_y2, bg_color):
    h, w = img_arr.shape[:2]
    bg_arr = np.array(bg_color)
    margin_x = max(20, (gpt_x2 - gpt_x1) // 5)
    margin_y = max(20, (gpt_y2 - gpt_y1) // 5)
    sx1 = max(0, gpt_x1 - margin_x)
    sy1 = max(0, gpt_y1 - margin_y)
    sx2 = min(w, gpt_x2 + margin_x)
    sy2 = min(h, gpt_y2 + margin_y)
    region = img_arr[sy1:sy2, sx1:sx2]
    diffs = np.abs(region.astype(int) - bg_arr).sum(axis=2)
    text_mask = diffs > 60
    if not text_mask.any():
        return gpt_x1, gpt_y1, gpt_x2, gpt_y2
    rows = np.any(text_mask, axis=1)
    cols = np.any(text_mask, axis=0)
    return (
        sx1 + int(np.where(cols)[0][0]),
        sy1 + int(np.where(rows)[0][0]),
        sx1 + int(np.where(cols)[0][-1]),
        sy1 + int(np.where(rows)[0][-1]),
    )

# ---------------------------------------------------------------
# 3. Funzione principale
# ---------------------------------------------------------------
def elabora_testo_dinamico(
    img_pil: Image.Image,
    openai_api_key: str,
    azione: str = "Rimuovi",
    new_text_str: str = "",
    font_file_bytes: bytes | None = None,
    font_size: int = 100,                    
    auto_scale: bool = False,                
    text_color_override: tuple | None = None 
) -> Image.Image:
    if azione == "Nessuna modifica":
        return img_pil

    img = img_pil.convert("RGB")
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]

    result = trova_anno_con_gpt(img, openai_api_key)
    if result is None:
        st.info("ℹ️ Nessun anno trovato — immagine lasciata intatta.")
        return img_pil

    gpt_x1 = int(result["x_pct"] * w / 100)
    gpt_y1 = int(result["y_pct"] * h / 100)
    gpt_x2 = min(w - 1, gpt_x1 + int(result["w_pct"] * w / 100))
    gpt_y2 = min(h - 1, gpt_y1 + int(result["h_pct"] * h / 100))

    left_x1 = max(0, gpt_x1 - 60)
    left_x2 = max(0, gpt_x1 - 5)
    band = img_arr[gpt_y1:gpt_y2, left_x1:left_x2] if left_x2 > left_x1 else img_arr[min(h-1, gpt_y2+5):min(h-1, gpt_y2+40), gpt_x1:gpt_x2]
    bg_color = tuple(int(v) for v in np.median(band.reshape(-1, 3), axis=0))

    x1, y1, x2, y2 = raffina_bbox_pixel(img_arr, gpt_x1, gpt_y1, gpt_x2, gpt_y2, bg_color)

    if text_color_override:
        text_color = text_color_override
    else:
        region = img_arr[y1:y2, x1:x2].reshape(-1, 3)
        bg_arr = np.array(bg_color)
        diffs = np.abs(region.astype(int) - bg_arr).sum(axis=1)
        text_pixels = region[diffs > 80]
        if len(text_pixels) > 0:
            text_color = tuple(int(v) for v in np.median(text_pixels, axis=0))
        else:
            brightness = sum(bg_color) / 3
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)

    padding = 6
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1 - padding, y1 - padding, x2 + padding, y2 + padding], fill=bg_color)

    if azione == "Sostituisci" and new_text_str:
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if font_file_bytes:
            if auto_scale:
                fs = font_size
                while fs > 8:
                    font = ImageFont.truetype(io.BytesIO(font_file_bytes), fs)
                    tb = font.getbbox(new_text_str)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                    if tw <= bbox_w and th <= bbox_h * 0.9:
                        break
                    fs -= 2
            else:
                font = ImageFont.truetype(io.BytesIO(font_file_bytes), font_size)
                tb = font.getbbox(new_text_str)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
        else:
            st.warning("⚠️ Nessun font caricato — uso font di sistema")
            font = ImageFont.load_default()
            tb = font.getbbox(new_text_str)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]

        draw_x = x2 - tw
        draw_y = y1 + (bbox_h - th) // 2
        draw.text((draw_x, draw_y), new_text_str, font=font, fill=text_color)

    return img


# ---------------------------------------------------------------
# 4. UI COMPLETA (APP STREAMLIT)
# ---------------------------------------------------------------
st.set_page_config(page_title="Editor Copertine", layout="wide")
st.title("Editor Copertine Fotolibri")

# Caricamento immagine base per far funzionare l'app
uploaded_file = st.file_uploader("Carica un'immagine di copertina", type=["jpg", "jpeg", "png"])

with st.expander("🤖 Sostituzione Anno con GPT-4o mini", expanded=True):
    
    # GESTIONE SECRETS
    if "OPENAI_API_KEY" in st.secrets:
        openai_key = st.secrets["OPENAI_API_KEY"]
        st.success("🔑 OpenAI API Key caricata automaticamente dai Secrets!")
    else:
        openai_key = st.text_input("🔑 OpenAI API Key", type="password")
        if not openai_key:
            st.warning("⚠️ Inserisci la chiave API per procedere.")

    modalita_testo = st.radio(
        "Azione:",
        ("Nessuna modifica", "Solo Rimuovi Anno", "Rimuovi e Sostituisci Anno"),
        index=0, horizontal=True
    )

    font_bytes = None
    new_text_input = ""
    font_size_input = 100
    auto_scale_input = False
    color_override = None
    usa_colore_auto = True

    if modalita_testo == "Rimuovi e Sostituisci Anno":
        col1, col2, col3 = st.columns(3)

        new_text_input = col1.text_input("Testo sostitutivo", value="2026")

        font_file = col2.file_uploader("Font (.ttf/.otf)", type=['ttf', 'otf'])
        if font_file:
            font_bytes = font_file.read()
            col2.success("✅ Font caricato!")
        else:
            col2.warning("Carica un font .ttf")

        font_size_input = col3.number_input("Dimensione font (px)", value=100, min_value=8, max_value=500)
        auto_scale_input = col3.checkbox("Scala automatico se non entra", value=False)

        # Colore testo: automatico o manuale
        usa_colore_auto = col1.checkbox("🎨 Colore automatico (rileva dall'originale)", value=True)
        if not usa_colore_auto:
            hex_color = col1.color_picker("Scegli colore testo", "#FFD700")
            hex_color = hex_color.lstrip('#')
            color_override = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ---------------------------------------------------------------
# ESECUZIONE
# ---------------------------------------------------------------
if uploaded_file is not None:
    d_img = Image.open(uploaded_file)
    
    if st.button("Elabora Immagine"):
        # Blocco di sicurezza se manca la chiave
        if not openai_key:
            st.error("❌ Impossibile procedere: manca la chiave API di OpenAI. Configurala nei secrets o inseriscila sopra.")
            st.stop()
            
        azione_passata = (
            "Sostituisci" if modalita_testo == "Rimuovi e Sostituisci Anno" else
            "Rimuovi"     if modalita_testo == "Solo Rimuovi Anno" else
            "Nessuna modifica"
        )

        with st.spinner("Analisi ed elaborazione in corso..."):
            img_processata = elabora_testo_dinamico(
                img_pil              = d_img,
                openai_api_key       = openai_key,
                azione               = azione_passata,
                new_text_str         = new_text_input,
                font_file_bytes      = font_bytes,
                font_size            = font_size_input,
                auto_scale           = auto_scale_input,
                text_color_override  = color_override if not usa_colore_auto else None
            )
        
        st.image(img_processata, caption="Immagine Risultante", use_container_width=True)
