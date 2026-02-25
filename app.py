import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, io, zipfile, re
import pytesseract

st.set_page_config(page_title="PhotoBook Mockup", layout="wide")

anno_pattern = re.compile(r'^20\d{2}$')

def trova_anno_tesseract(img_pil):
    img_arr = np.array(img_pil.convert('RGB'))
    h, w = img_arr.shape[:2]
    zona_h = int(h * 0.55)
    zona = img_arr[:zona_h, :, :]
    bg_color = np.median(img_arr[5:25, 5:25].reshape(-1, 3), axis=0)
    pixels = zona.reshape(-1, 3).astype(int)
    diff_bg = np.abs(pixels - bg_color).sum(axis=1)
    text_pixels = pixels[diff_bg > 80]
    if len(text_pixels) == 0:
        return None
    rounded = (text_pixels // 30 * 30)
    unique, counts = np.unique(rounded, axis=0, return_counts=True)
    top_colors = unique[np.argsort(-counts)[:8]]
    best = None
    best_conf = -1
    for color in top_colors:
        color_diff = np.abs(img_arr[:zona_h].astype(int) - color).sum(axis=2)
        mask = np.where(color_diff < 55, 0, 255).astype(np.uint8)
        img_mask = Image.fromarray(mask)
        scale = 3
        img_big = img_mask.resize((w*scale, zona_h*scale), Image.LANCZOS)
        try:
            data = pytesseract.image_to_data(img_big, output_type=pytesseract.Output.DICT,
                                              config='--psm 11 -c tessedit_char_whitelist=0123456789')
            for i, text in enumerate(data['text']):
                t = text.strip()
                conf = int(data['conf'][i])
                if anno_pattern.match(t) and conf > 50:
                    x1 = data['left'][i] // scale
                    y1 = data['top'][i] // scale
                    bw = data['width'][i] // scale
                    bh = data['height'][i] // scale
                    if conf > best_conf:
                        best_conf = conf
                        best = (x1, y1, x1+bw, y1+bh, tuple(int(v) for v in color))
        except:
            pass
    return best


def elabora_testo_dinamico(img_pil, azione="Rimuovi",
                            new_text_str="", font_file_bytes=None,
                            font_size=None, text_color_override=None):
    if azione == "Nessuna modifica":
        return img_pil
    result = trova_anno_tesseract(img_pil)
    if result is None:
        st.info("ℹ️ Anno non trovato — immagine lasciata intatta.")
        return img_pil
    x1, y1, x2, y2, auto_text_color = result
    img = img_pil.convert("RGB")
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    bbox_h = y2 - y1
    pad = 30
    ys1, ys2 = max(0, y1-pad), min(h, y2+pad)
    xs1, xs2 = max(0, x1-pad), min(w, x2+pad)
    surround = img_arr[ys1:ys2, xs1:xs2].copy()
    mask = np.ones(surround.shape[:2], dtype=bool)
    mask[y1-ys1:y2-ys1, x1-xs1:x2-xs1] = False
    bg_pixels = surround[mask]
    bg_color = tuple(int(v) for v in np.median(bg_pixels, axis=0))
    text_color = text_color_override if text_color_override else auto_text_color
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1-25, y1-15, x2+25, y2+15], fill=bg_color)
    if azione == "Sostituisci" and new_text_str and font_file_bytes:
        try:
            if font_size is None:
                fs = 8
                while fs < 500:
                    font = ImageFont.truetype(io.BytesIO(font_file_bytes), fs)
                    tb = font.getbbox(new_text_str)
                    if (tb[3]-tb[1]) >= bbox_h * 0.85: break
                    fs += 1
            else:
                fs = font_size
            font = ImageFont.truetype(io.BytesIO(font_file_bytes), fs)
            tb = font.getbbox(new_text_str)
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
            draw_x = x2 - tw
            draw_y = y1 + (bbox_h - th) // 2
            draw.text((draw_x, draw_y), new_text_str, font=font, fill=text_color)
        except Exception as e:
            st.warning(f"Errore font: {e}")
    return img


# --- UI ---
st.title("📸 PhotoBook Anno Replacer")

with st.expander("⚙️ Impostazioni sostituzione anno", expanded=True):
    modalita_testo = st.radio(
        "Azione:",
        ("Nessuna modifica", "Solo Rimuovi Anno", "Rimuovi e Sostituisci Anno"),
        horizontal=True
    )
    font_bytes = None
    new_text_input = ""
    font_size_input = None
    color_override = None
    usa_colore_auto = True

    if modalita_testo == "Rimuovi e Sostituisci Anno":
        col1, col2, col3 = st.columns(3)
        new_text_input = col1.text_input("Testo sostitutivo", value="2026")
        font_file = col2.file_uploader("Font (.ttf/.otf)", type=['ttf','otf'])
        if font_file:
            font_bytes = font_file.read()
            col2.success("✅ Font caricato!")
        else:
            col2.warning("Carica un font .ttf")
        usa_auto = col3.checkbox("🔁 Dimensione automatica", value=True)
        if not usa_auto:
            font_size_input = col3.number_input("Dimensione font (px)", value=80, min_value=8, max_value=500)
        usa_colore_auto = col1.checkbox("🎨 Colore automatico", value=True)
        if not usa_colore_auto:
            hex_c = col1.color_picker("Colore testo", "#FFD700")
            hex_c = hex_c.lstrip('#')
            color_override = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))

st.divider()

def processa_img(img_pil):
    if modalita_testo == "Nessuna modifica":
        return img_pil
    azione = "Sostituisci" if modalita_testo == "Rimuovi e Sostituisci Anno" else "Rimuovi"
    return elabora_testo_dinamico(
        img_pil=img_pil, azione=azione,
        new_text_str=new_text_input,
        font_file_bytes=font_bytes,
        font_size=font_size_input,
        text_color_override=color_override if not usa_colore_auto else None
    )

# Anteprima singola
st.subheader("🔍 Anteprima singola")
up = st.file_uploader("Carica un design", type=['jpg','png','jpeg'], key='preview')
if up:
    d_img = Image.open(up)
    d_img = processa_img(d_img)
    st.image(d_img, caption="Design Processato", width=300)
    buf_dl = io.BytesIO()
    ext = 'png' if up.name.lower().endswith('.png') else 'jpg'
    if ext == 'png': d_img.save(buf_dl, format='PNG')
    else: d_img.save(buf_dl, format='JPEG', quality=95)
    st.download_button("📥 Scarica", buf_dl.getvalue(),
                       f"{os.path.splitext(up.name)[0]}_processato.{ext}")

st.divider()

# Batch
st.subheader("🚀 Batch")
batch = st.file_uploader("Carica tutti i design", accept_multiple_files=True, type=['jpg','png','jpeg'], key='batch')
if st.button("🚀 PROCESSA BATCH") and batch:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
        progress = st.progress(0)
        for idx, b_file in enumerate(batch):
            b_img = Image.open(b_file)
            b_img = processa_img(b_img)
            ext = 'png' if b_file.name.lower().endswith('.png') else 'jpg'
            buf2 = io.BytesIO()
            if ext == 'png': b_img.save(buf2, format='PNG')
            else: b_img.save(buf2, format='JPEG', quality=95)
            zf.writestr(f"{os.path.splitext(b_file.name)[0]}_processato.{ext}", buf2.getvalue())
            progress.progress((idx+1)/len(batch))
    st.session_state.zip_ready = True
    st.session_state.zip_data = zip_buf.getvalue()
    st.success(f"✅ {len(batch)} design processati!")

if st.session_state.get('zip_ready'):
    st.download_button("📥 SCARICA ZIP", st.session_state.zip_data, "design_processati.zip", "application/zip")
