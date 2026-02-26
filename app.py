import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os, io, zipfile, re
import pytesseract

st.set_page_config(page_title="PhotoBook Anno Replacer", layout="wide")

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

anno_pattern = re.compile(r'^20\d{2}$')

def trova_bordi_cover(img_arr):
    h, w = img_arr.shape[:2]
    corners = [img_arr[3,3], img_arr[3,w-3], img_arr[h-3,3], img_arr[h-3,w-3]]
    outer_bg = np.median(corners, axis=0)
    diff = np.abs(img_arr.astype(int) - outer_bg).sum(axis=2)
    inside = diff > 15
    cols = np.where(np.any(inside, axis=0))[0]
    rows = np.where(np.any(inside, axis=1))[0]
    if len(cols) > 0 and len(rows) > 0:
        return int(cols[0])+2, int(rows[0])+2, int(cols[-1])-2, int(rows[-1])-2
    return 0, 0, w-1, h-1

def trova_anno(img_pil):
    img_arr = np.array(img_pil.convert('RGB'))
    h, w = img_arr.shape[:2]
    zona_h = int(h * 0.52)

    # Sfondo: zona centrale sinistra
    bg_color = np.median(
        img_arr[int(h*0.25):int(h*0.48), int(w*0.05):int(w*0.3)].reshape(-1,3), axis=0)

    best = None
    best_conf = -1

    # PASSO 1: Tesseract con isolamento colore
    pixels = img_arr[:zona_h].reshape(-1,3).astype(int)
    diff_bg = np.abs(pixels - bg_color).sum(axis=1)
    text_pixels = pixels[diff_bg > 20]

    if len(text_pixels) > 0:
        rounded = (text_pixels // 20 * 20)
        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        top_colors = unique[np.argsort(-counts)[:15]]

        for color in top_colors:
            color_diff = np.abs(img_arr[:zona_h].astype(int) - color).sum(axis=2)
            mask = np.where(color_diff < 40, 0, 255).astype(np.uint8)
            img_big = Image.fromarray(mask).resize((w*3, zona_h*3), Image.LANCZOS)
            try:
                data = pytesseract.image_to_data(img_big, output_type=pytesseract.Output.DICT,
                                                  config='--psm 11 -c tessedit_char_whitelist=0123456789')
                for i, text in enumerate(data['text']):
                    t = text.strip()
                    conf = int(data['conf'][i])
                    if anno_pattern.match(t) and conf > 30:
                        x1 = data['left'][i]//3; y1 = data['top'][i]//3
                        x2 = x1+data['width'][i]//3; y2 = y1+data['height'][i]//3
                        if y1 < h*0.12: continue  # non sovrapporre titolo
                        if conf > best_conf:
                            best_conf = conf
                            best = (x1, y1, x2, y2, tuple(int(v) for v in color))
            except: pass

    # PASSO 2: Tesseract ad alto contrasto
    if best is None:
        zona_destra = img_pil.crop((int(w*0.35), int(h*0.12), w, zona_h)).convert('RGB')
        enhanced = ImageEnhance.Contrast(zona_destra).enhance(4.0)
        img_big2 = enhanced.resize((zona_destra.width*4, zona_destra.height*4), Image.LANCZOS)
        try:
            data = pytesseract.image_to_data(img_big2, output_type=pytesseract.Output.DICT,
                                              config='--psm 11 -c tessedit_char_whitelist=0123456789')
            for i, text in enumerate(data['text']):
                t = text.strip()
                conf = int(data['conf'][i])
                if anno_pattern.match(t) and conf > 20:
                    ox = int(w*0.35); oy = int(h*0.12)
                    x1 = ox+data['left'][i]//4; y1 = oy+data['top'][i]//4
                    x2 = x1+data['width'][i]//4; y2 = y1+data['height'][i]//4
                    if conf > best_conf:
                        best_conf = conf
                        # campiona colore testo nel bbox
                        region = img_arr[y1:y2, x1:x2].reshape(-1,3)
                        diffs = np.abs(region.astype(int) - bg_color).sum(axis=1)
                        tp = region[diffs > 20]
                        tc = tuple(int(v) for v in np.median(tp, axis=0)) if len(tp)>0 else (50,50,50)
                        best = (x1, y1, x2, y2, tc)
        except: pass

    # PASSO 3: fallback zona fissa — MA bbox limitato a dimensione realistica
    if best is None:
        fx1, fy1 = int(w*0.42), int(h*0.16)
        fx2, fy2 = int(w*0.95), int(h*0.38)
        region = img_arr[fy1:fy2, fx1:fx2].reshape(-1,3)
        diffs = np.abs(region.astype(int) - bg_color).sum(axis=1)
        tp = region[diffs > 20]
        tc = tuple(int(v) for v in np.median(tp, axis=0)) if len(tp)>0 else (50,50,50)
        # Raffina bbox pixel
        cdiff = np.abs(img_arr[fy1:fy2, fx1:fx2].astype(int) - np.array(tc)).sum(axis=2)
        tmask = cdiff < 35
        if tmask.any():
            rows2 = np.where(np.any(tmask, axis=1))[0]
            cols2 = np.where(np.any(tmask, axis=0))[0]
            rx1 = fx1+int(cols2[0]); ry1 = fy1+int(rows2[0])
            rx2 = fx1+int(cols2[-1]); ry2 = fy1+int(rows2[-1])
            # Solo se bbox ragionevole
            if (rx2-rx1) > 20 and (ry2-ry1) > 10 and (ry2-ry1) < h*0.2:
                best = (rx1, ry1, rx2, ry2, tc)
            else:
                best = (fx1, fy1, fx2, fy2, tc)
        else:
            best = (fx1, fy1, fx2, fy2, tc)

    # Sanity: bbox non deve essere più alto del 15% dell'immagine
    if best:
        x1,y1,x2,y2,tc = best
        if (y2-y1) > h*0.15:
            # Ritaglia al centro verticale del bbox con altezza max 10%
            mid_y = (y1+y2)//2
            new_h = int(h*0.08)
            best = (x1, mid_y-new_h//2, x2, mid_y+new_h//2, tc)

    return best


def elabora_testo_dinamico(img_pil, azione="Rimuovi",
                            new_text_str="", font_file_bytes=None,
                            font_size=None, text_color_override=None):
    if azione == "Nessuna modifica":
        return img_pil

    result = trova_anno(img_pil)
    if result is None:
        st.info("ℹ️ Anno non trovato.")
        return img_pil

    x1, y1, x2, y2, auto_text_color = result
    img = img_pil.convert("RGB")
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    bbox_h = y2 - y1

    cx1, cy1, cx2, cy2 = trova_bordi_cover(img_arr)

    # Sfondo intorno al bbox
    pad = 30
    ys1,ys2 = max(0,y1-pad), min(h,y2+pad)
    xs1,xs2 = max(0,x1-pad), min(w,x2+pad)
    surround = img_arr[ys1:ys2, xs1:xs2].copy()
    mask = np.ones(surround.shape[:2], dtype=bool)
    mask[y1-ys1:y2-ys1, x1-xs1:x2-xs1] = False
    bg_pixels = surround[mask]
    bg_color = tuple(int(v) for v in np.median(bg_pixels, axis=0)) if len(bg_pixels)>0 else (255,255,255)

    text_color = text_color_override if text_color_override else auto_text_color

    draw = ImageDraw.Draw(img)
    draw.rectangle([max(cx1,x1-25), max(cy1,y1-15),
                    min(cx2,x2+25), min(cy2,y2+15)], fill=bg_color)

    if azione == "Sostituisci" and new_text_str and font_file_bytes:
        try:
            if font_size is None:
                max_h = min(bbox_h, 65)  # mai più alto di 65px
                fs = 8
                while fs < 300:
                    font = ImageFont.truetype(io.BytesIO(font_file_bytes), fs)
                    tb = font.getbbox(new_text_str)
                    if (tb[3]-tb[1]) >= max_h * 0.85: break
                    fs += 1
            else:
                fs = font_size
            font = ImageFont.truetype(io.BytesIO(font_file_bytes), fs)
            tb = font.getbbox(new_text_str)
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
            draw_x = x2 - tw
            draw_y = y1 + (bbox_h - th) // 2 - 35
            draw.text((draw_x, draw_y), new_text_str, font=font, fill=text_color)
        except Exception as e:
            st.warning(f"Errore font: {e}")

    return img


# --- UI ---
st.title("📸 PhotoBook Anno Replacer")

with st.expander("⚙️ Impostazioni", expanded=True):
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
            font_size_input = col3.number_input("Dimensione font (px)", value=55, min_value=8, max_value=300)
        usa_colore_auto = col1.checkbox("🎨 Colore automatico", value=True)
        if not usa_colore_auto:
            hex_c = col1.color_picker("Colore testo", "#FFD700")
            hex_c = hex_c.lstrip('#')
            color_override = tuple(int(hex_c[i:i+2], 16) for i in (0,2,4))

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

st.subheader("🚀 Batch")
batch = st.file_uploader("Carica tutti i design", accept_multiple_files=True,
                          type=['jpg','png','jpeg'], key='batch')
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
    st.download_button("📥 SCARICA ZIP", st.session_state.zip_data,
                       "design_processati.zip", "application/zip")
