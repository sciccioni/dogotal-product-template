import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, io, zipfile, re
import pytesseract

st.set_page_config(page_title="PhotoBook Mockup", layout="wide")

anno_pattern = re.compile(r'^20\d{2}$')

def trova_anno_tesseract(img_pil):
    """
    Cerca l'anno con Tesseract isolando ogni colore dominante.
    Se non trova nulla, usa zona fissa di fallback (destra, 18%-40% altezza).
    Ritorna (x1, y1, x2, y2, text_color) o None.
    """
    img_arr = np.array(img_pil.convert('RGB'))
    h, w = img_arr.shape[:2]
    zona_h = int(h * 0.55)

    # Sfondo: campiona zona centrale sinistra (sicuramente sfondo)
    bg_sample = img_arr[int(h*0.3):int(h*0.5), int(w*0.05):int(w*0.25)]
    bg_color = np.median(bg_sample.reshape(-1, 3), axis=0)

    # Trova colori dominanti non-sfondo
    pixels = img_arr[:zona_h].reshape(-1, 3).astype(int)
    diff_bg = np.abs(pixels - bg_color).sum(axis=1)
    text_pixels = pixels[diff_bg > 25]

    best = None
    best_conf = -1

    if len(text_pixels) > 0:
        rounded = (text_pixels // 25 * 25)
        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        top_colors = unique[np.argsort(-counts)[:12]]

        for color in top_colors:
            color_diff = np.abs(img_arr[:zona_h].astype(int) - color).sum(axis=2)
            mask = np.where(color_diff < 45, 0, 255).astype(np.uint8)
            img_mask = Image.fromarray(mask)
            scale = 3
            img_big = img_mask.resize((w*scale, zona_h*scale), Image.LANCZOS)
            try:
                data = pytesseract.image_to_data(img_big, output_type=pytesseract.Output.DICT,
                                                  config='--psm 11 -c tessedit_char_whitelist=0123456789')
                for i, text in enumerate(data['text']):
                    t = text.strip()
                    conf = int(data['conf'][i])
                    if anno_pattern.match(t) and conf > 30:
                        x1 = data['left'][i] // scale
                        y1 = data['top'][i] // scale
                        bw = data['width'][i] // scale
                        bh = data['height'][i] // scale
                        if conf > best_conf:
                            best_conf = conf
                            best = (x1, y1, x1+bw, y1+bh, tuple(int(v) for v in color))
            except:
                pass

    # Fallback: zona fissa destra sotto il titolo
    if best is None:
        fx1 = int(w * 0.45)
        fy1 = int(h * 0.18)
        fx2 = int(w * 0.95)
        fy2 = int(h * 0.40)
        # Colore testo: pixel nella zona fallback più diversi dallo sfondo
        region = img_arr[fy1:fy2, fx1:fx2].reshape(-1, 3)
        diffs = np.abs(region.astype(int) - bg_color).sum(axis=1)
        text_pix = region[diffs > 20]
        if len(text_pix) > 0:
            rounded2 = (text_pix // 20 * 20)
            u2, c2 = np.unique(rounded2, axis=0, return_counts=True)
            # Prendi il colore più comune che NON è lo sfondo
            text_color_fb = tuple(int(v) for v in u2[np.argmax(c2)])
        else:
            # Testo bianco se sfondo scuro, nero se sfondo chiaro
            brightness = sum(int(v) for v in bg_color) / 3
            text_color_fb = (255,255,255) if brightness < 128 else (50,50,50)
        best = (fx1, fy1, fx2, fy2, text_color_fb)
        best = (best[0], best[1], best[2], best[3], best[4])
        # Raffina il bbox con scansione pixel nella zona fallback
        color_diff2 = np.abs(img_arr[fy1:fy2, fx1:fx2].astype(int) - np.array(text_color_fb)).sum(axis=2)
        text_mask2 = color_diff2 < 40
        if text_mask2.any():
            rows2 = np.any(text_mask2, axis=1)
            cols2 = np.any(text_mask2, axis=0)
            ys2 = np.where(rows2)[0]
            xs2 = np.where(cols2)[0]
            best = (fx1 + int(xs2[0]), fy1 + int(ys2[0]),
                    fx1 + int(xs2[-1]), fy1 + int(ys2[-1]),
                    text_color_fb)

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

    # Trova i bordi reali della cover (esclude trasparenza/sfondo bianco esterno)
    # Campiona colori ai 4 angoli per trovare sfondo esterno
    corners = [img_arr[3,3], img_arr[3,w-3], img_arr[h-3,3], img_arr[h-3,w-3]]
    outer_bg = np.median(corners, axis=0)
    # Mappa binaria: 1 = dentro la cover, 0 = fuori
    outer_diff = np.abs(img_arr.astype(int) - outer_bg).sum(axis=2)
    inside_cover = outer_diff > 15
    # Trova bordi cover per riga (x1_cover, x2_cover per ogni y)
    cover_cols = np.where(np.any(inside_cover, axis=0))[0]
    cover_rows = np.where(np.any(inside_cover, axis=1))[0]
    if len(cover_cols) > 0 and len(cover_rows) > 0:
        cx1, cx2 = int(cover_cols[0])+2, int(cover_cols[-1])-2
        cy1, cy2 = int(cover_rows[0])+2, int(cover_rows[-1])-2
    else:
        cx1, cy1, cx2, cy2 = 0, 0, w-1, h-1

    # Colore sfondo: campiona zona centrale sinistra (dentro la cover)
    bg_sample = img_arr[int(h*0.3):int(h*0.5), cx1+5:cx1+int(w*0.2)]
    bg_color = tuple(int(v) for v in np.median(bg_sample.reshape(-1,3), axis=0))
    text_color = text_color_override if text_color_override else auto_text_color

    # Cancella — clampato ai bordi reali della cover
    ex1 = max(cx1, x1-25)
    ey1 = max(cy1, y1-15)
    ex2 = min(cx2, x2+25)
    ey2 = min(cy2, y2+15)
    draw = ImageDraw.Draw(img)
    draw.rectangle([ex1, ey1, ex2, ey2], fill=bg_color)
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
