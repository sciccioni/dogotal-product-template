import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
import io
import zipfile
import json

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="PhotoBook Mockup Compositor - V5 SMART", layout="wide")

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- ANTI-CACHE ---
def get_folder_hash(folder_path):
    if not os.path.exists(folder_path):
        return 0
    return sum(os.path.getmtime(os.path.join(folder_path, f)) for f in os.listdir(folder_path))

# --- COORDINATE ---
TEMPLATE_MAPS_FILE = "template_coordinates.json"

def load_template_maps():
    default_maps = {
        "base_verticale_temi_app.jpg": {"coords": (34.4, 9.1, 30.6, 80.4), "offset": 1},
        "base_orizzontale_temi_app.jpg": {"coords": (18.9, 9.4, 61.9, 83.0), "offset": 1},
        "base_orizzontale_temi_app3.jpg": {"coords": (18.7, 9.4, 62.2, 82.9), "offset": 1},
        "base_quadrata_temi_app.jpg": {"coords": (27.7, 10.5, 44.7, 79.4), "offset": 1},
        "base_bottom_app.jpg": {"coords": (21.8, 4.7, 57.0, 91.7), "offset": 1},
        "15x22-crea la tua grafica.jpg": {"coords": (33.1, 21.4, 33.9, 57.0), "offset": 2},
        "20x30-crea la tua grafica.jpg": {"coords": (33.1, 21.4, 33.9, 57.0), "offset": 2},
        "Fotolibro-Temi-Verticali-temi-2.png": {"coords": (13.6, 4.0, 73.0, 92.0), "offset": 1},
        "Fotolibro-Temi-Verticali-temi-3.png": {"coords": (13.6, 4.0, 73.0, 92.0), "offset": 1}
    }
    if os.path.exists(TEMPLATE_MAPS_FILE):
        try:
            with open(TEMPLATE_MAPS_FILE, 'r') as f:
                return json.load(f)
        except:
            return default_maps
    return default_maps

def save_template_maps(maps):
    with open(TEMPLATE_MAPS_FILE, 'w') as f:
        json.dump(maps, f, indent=2)

TEMPLATE_MAPS = load_template_maps()

# --- CATEGORIE ---
def get_manual_cat(filename):
    fn = filename.lower()
    if any(x in fn for x in ["vertical", "15x22", "20x30", "bottom", "copertina_verticale"]):
        return "Verticali"
    if any(x in fn for x in ["orizzontal", "20x15", "27x20", "32x24", "40x30"]):
        return "Orizzontali"
    if any(x in fn for x in ["quadrat", "20x20", "30x30"]):
        return "Quadrati"
    return "Altro"

# --- CORE LOGIC MOCKUP ---
def find_book_region(tmpl_gray, bg_val):
    h, w = tmpl_gray.shape
    book_mask = tmpl_gray > (bg_val + 3)
    rows, cols = np.any(book_mask, axis=1), np.any(book_mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    by1, by2 = np.where(rows)[0][[0, -1]]
    bx1, bx2 = np.where(cols)[0][[0, -1]]
    mid_y = (by1 + by2) // 2
    row = tmpl_gray[mid_y]
    face_x1 = bx1
    for x in range(bx1, bx2 - 5):
        if np.all(row[x:x+5] >= 240):
            face_x1 = x
            break
    return {
        'book_x1': int(bx1), 'book_x2': int(bx2),
        'book_y1': int(by1), 'book_y2': int(by2),
        'face_x1': int(face_x1)
    }

def composite_v3_fixed(tmpl_pil, cover_pil, template_name="", border_offset=None):
    has_alpha = False
    alpha_mask = None
    if (tmpl_pil.mode in ('RGBA', 'LA') or
            (tmpl_pil.mode == 'P' and 'transparency' in tmpl_pil.info) or
            template_name.lower().endswith('.png')):
        has_alpha = True
        tmpl_pil = tmpl_pil.convert('RGBA')
        alpha_mask = tmpl_pil.split()[3]

    tmpl_rgb = tmpl_pil.convert('RGB')
    h, w = tmpl_rgb.size[1], tmpl_rgb.size[0]

    if template_name in TEMPLATE_MAPS:
        d = TEMPLATE_MAPS[template_name]
        px, py, pw, ph = d["coords"]
        bo = border_offset if border_offset is not None else d.get("offset", 1)
        x1, y1 = int((px * w) / 100) + bo, int((py * h) / 100) + bo
        tw, th = int((pw * w) / 100) - (bo * 2), int((ph * h) / 100) - (bo * 2)
        target_aspect = tw / th
        cw, ch = cover_pil.size
        if cw/ch > target_aspect:
            nw = int(ch * target_aspect)
            crop = ((cw - nw)//2, 0, (cw - nw)//2 + nw, ch)
        else:
            nh = int(cw / target_aspect)
            crop = (0, (ch - nh)//2, cw, (ch - nh)//2 + nh)
        c_res = cover_pil.crop(crop).resize((tw, th), Image.LANCZOS)
        tmpl_l = np.array(tmpl_rgb.convert('L')).astype(np.float64)
        shadows = np.clip(tmpl_l[y1:y1+th, x1:x1+tw] / 246.0, 0, 1.0)
        c_array = np.array(c_res.convert('RGB')).astype(np.float64)
        for i in range(3):
            c_array[:,:,i] *= shadows
        final_face = Image.fromarray(c_array.astype(np.uint8))
        if c_res.mode == 'RGBA':
            tmpl_rgb.paste(final_face, (x1, y1), c_res)
        else:
            tmpl_rgb.paste(final_face, (x1, y1))
        if has_alpha:
            tmpl_rgb.putalpha(alpha_mask)
        return tmpl_rgb

    tmpl_gray = np.array(tmpl_rgb.convert('L'))
    corners = [tmpl_gray[3,3], tmpl_gray[3,w-3], tmpl_gray[h-3,3], tmpl_gray[h-3,w-3]]
    bg_val = float(np.median(corners))
    region = find_book_region(tmpl_gray, bg_val)
    if region is None or (region['book_x2'] - region['book_x1'] > w * 0.95):
        margin_x, margin_y = int(w * 0.2), int(h * 0.1)
        bx1, bx2, by1, by2 = margin_x, w - margin_x, margin_y, h - margin_y
    else:
        bx1, bx2, by1, by2 = region['book_x1'], region['book_x2'], region['book_y1'], region['book_y2']

    if "base_copertina" in template_name.lower():
        bx1, bx2, by1, by2 = 0, w-1, 0, h-1
        for x in range(w):
            if np.any(tmpl_gray[:, x] < 250):
                bx1 = x
                break
        for x in range(w-1, -1, -1):
            if np.any(tmpl_gray[:, x] < 250):
                bx2 = x
                break
        for y in range(h):
            if np.any(tmpl_gray[y, :] < 250):
                by1 = y
                break
        for y in range(h-1, -1, -1):
            if np.any(tmpl_gray[y, :] < 250):
                by2 = y
                break

    bx1, bx2 = max(0, bx1 - 2), min(w - 1, bx2 + 2)
    by1, by2 = max(0, by1 - 2), min(h - 1, by2 + 2)
    tw, th = bx2 - bx1 + 1, by2 - by1 + 1
    c_res = cover_pil.resize((tw, th), Image.LANCZOS)
    c_arr = np.array(c_res.convert('RGB')).astype(np.float64)
    sh = np.clip(tmpl_gray[by1:by2+1, bx1:bx2+1] / 246.0, 0, 1.0)
    for i in range(3):
        c_arr[:,:,i] *= sh
    final_face = Image.fromarray(c_arr.astype(np.uint8))
    if c_res.mode == 'RGBA':
        tmpl_rgb.paste(final_face, (bx1, by1), c_res)
    else:
        tmpl_rgb.paste(final_face, (bx1, by1))
    if has_alpha:
        tmpl_rgb.putalpha(alpha_mask)
    return tmpl_rgb

# --- NUOVA FUNZIONE: RIMOZIONE TESTO SMART ---
def applica_censura_smart(img_pil, search_x1, search_x2, search_y1, search_y2, sample_x, sample_y, padding=8):
    """
    Cerca la scritta all'interno di una 'Zona di Ricerca' e la copre dinamicamente.
    """
    img = img_pil.convert("RGB")
    img_arr = np.array(img)
    h, w, _ = img_arr.shape
    
    # 1. Trova il colore di sfondo per QUESTA immagine
    bg_color = img_arr[sample_y, sample_x]
    
    # 2. Converti le percentuali in coordinate pixel per la zona di ricerca
    sx1, sy1 = int((search_x1 * w) / 100), int((search_y1 * h) / 100)
    sx2, sy2 = int((search_x2 * w) / 100), int((search_y2 * h) / 100)
    
    # 3. Estrai l'area di ricerca
    search_zone = img_arr[sy1:sy2, sx1:sx2]
    
    # Se la zona di ricerca è vuota/invalida, ritorna l'immagine originale
    if search_zone.size == 0:
        return img
        
    # 4. Trova i pixel DIVERSI dal colore di sfondo (tolleranza per artefatti JPEG)
    diff = np.abs(search_zone.astype(int) - bg_color.astype(int))
    mask = np.sum(diff, axis=2) > 40  # 40 è la tolleranza. Se il pixel differisce, è "testo"
    
    # Se non c'è testo diverso dallo sfondo, ritorna l'immagine intatta
    if not np.any(mask):
        return img
        
    # 5. Calcola le coordinate esatte della scritta trovata
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # 6. Riporta le coordinate in scala globale (aggiungendo un po' di margine/padding)
    final_x1 = max(0, sx1 + xmin - padding)
    final_y1 = max(0, sy1 + ymin - padding)
    final_x2 = min(w, sx1 + xmax + padding)
    final_y2 = min(h, sy1 + ymax + padding)
    
    # 7. Disegna la toppa esatta
    draw = ImageDraw.Draw(img)
    draw.rectangle([final_x1, final_y1, final_x2, final_y2], fill=tuple(bg_color))
    
    return img

# --- LIBRERIA ---
@st.cache_data
def get_lib(h_val):
    lib = {"Verticali": {}, "Orizzontali": {}, "Quadrati": {}, "Altro": {}}
    if not os.path.exists("templates"):
        return lib
    for f in os.listdir("templates"):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            lib[get_manual_cat(f)][f] = Image.open(os.path.join("templates", f))
    return lib

libreria = get_lib(get_folder_hash("templates"))

# --- INTERFACCIA ---
menu = st.sidebar.radio("Menu", ["📚 Templates", "🎯 Calibrazione", "⚡ Produzione"])

if menu == "📚 Templates":
    st.subheader("📚 Libreria Templates")
    if st.button("🔄 RICARICA"):
        st.cache_data.clear()
        st.rerun()
    ts = st.tabs(list(libreria.keys()))
    for i, c in enumerate(libreria.keys()):
        with ts[i]:
            cols = st.columns(4)
            for idx, (fn, img) in enumerate(libreria[c].items()):
                cols[idx%4].image(img, caption=fn, use_column_width=True)

elif menu == "🎯 Calibrazione":
    cat = st.selectbox("Categoria:", list(libreria.keys()))
    sel = st.selectbox("Template:", list(libreria[cat].keys()))
    if sel:
        t_img = libreria[cat][sel]
        d = TEMPLATE_MAPS.get(sel, {"coords": (20, 10, 60, 80), "offset": 1})
        if 'cal' not in st.session_state or st.session_state.get('cur') != sel:
            st.session_state.cal = d
            st.session_state.cur = sel
        c = list(st.session_state.cal["coords"])
        st.session_state.cal["coords"] = c
        col1, col2 = st.columns(2)
        c[0] = col1.number_input("X %", 0.0, 100.0, float(c[0]))
        c[1] = col2.number_input("Y %", 0.0, 100.0, float(c[1]))
        c[2] = col1.number_input("W %", 0.0, 100.0, float(c[2]))
        c[3] = col2.number_input("H %", 0.0, 100.0, float(c[3]))
        st.session_state.cal["offset"] = st.slider("Offset", 0, 20, int(st.session_state.cal["offset"]))
        p_img = t_img.copy().convert('RGB')
        draw = ImageDraw.Draw(p_img)
        w, h = p_img.size
        draw.rectangle(
            [int(c[0]*w/100), int(c[1]*h/100), int((c[0]+c[2])*w/100), int((c[1]+c[3])*h/100)],
            outline="red", width=5
        )
        st.image(p_img, use_column_width=True)
        if st.button("💾 SALVA"):
            TEMPLATE_MAPS[sel] = st.session_state.cal
            save_template_maps(TEMPLATE_MAPS)
            st.success("Salvate!")

elif menu == "⚡ Produzione":
    scelta = st.radio("Formato:", ["Verticali", "Orizzontali", "Quadrati"], horizontal=True)
    
    # --- OPZIONI CENSURA ---
    with st.expander("🖌️ Auto-Rimozione Testo (Insegue la scritta)", expanded=False):
        st.write("Imposta una **Zona di Ricerca** ampia (il riquadro in cui solitamente appare il '2025'). Lo script cercherà la scritta lì dentro e la cancellerà in base alle sue dimensioni reali.")
        usa_censura = st.checkbox("Attiva auto-rimozione per questi caricamenti")
        if usa_censura:
            col_cen1, col_cen2 = st.columns(2)
            # Area di ricerca bella ampia per la tua immagine (prende metà destra)
            cens_x1 = col_cen1.slider("Ricerca Inizio X (%)", 0, 100, 50)
            cens_x2 = col_cen1.slider("Ricerca Fine X (%)", 0, 100, 95)
            cens_y1 = col_cen2.slider("Ricerca Inizio Y (%)", 0, 100, 20)
            cens_y2 = col_cen2.slider("Ricerca Fine Y (%)", 0, 100, 45)
            
            samp_x = st.number_input("Pixel Colore Sfondo X (es: 10)", 0, 10000, 10)
            samp_y = st.number_input("Pixel Colore Sfondo Y (es: 10)", 0, 10000, 10)

    st.divider()

    up = st.file_uploader("Carica design singolo (Anteprima)", type=['jpg', 'png'], key='preview')
    if up and libreria[scelta]:
        d_img = Image.open(up)
        
        if usa_censura:
            d_img = applica_censura_smart(d_img, cens_x1, cens_x2, cens_y1, cens_y2, samp_x, samp_y)
            st.info("Anteprima: La scritta è stata localizzata e rimossa!")
            st.image(d_img, caption="Design Pulito", width=250)

        cols = st.columns(4)
        for i, (t_name, t_img) in enumerate(libreria[scelta].items()):
            with cols[i%4]:
                res = composite_v3_fixed(t_img, d_img, t_name)
                st.image(res, caption=t_name, use_column_width=True)
    
    st.divider()
    
    batch = st.file_uploader("Batch Produzione (Zippa tutto)", accept_multiple_files=True)
    if st.button("🚀 GENERA TUTTI (BATCH)") and batch and libreria[scelta]:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
            progress = st.progress(0)
            total = len(batch) * len(libreria[scelta])
            count = 0
            for b_file in batch:
                b_img = Image.open(b_file)
                
                # CENSURA DINAMICA NEL BATCH
                if usa_censura:
                    b_img = applica_censura_smart(b_img, cens_x1, cens_x2, cens_y1, cens_y2, samp_x, samp_y)

                base_name = os.path.splitext(b_file.name)[0]
                if base_name.lower().endswith('.png'):
                    base_name = base_name[:-4]
                
                for t_name, t_img in libreria[scelta].items():
                    res = composite_v3_fixed(t_img, b_img, t_name)
                    is_tmpl_png = t_name.lower().endswith('.png') or res.mode == 'RGBA'
                    save_fmt = 'PNG' if is_tmpl_png else 'JPEG'
                    save_ext = '.png' if is_tmpl_png else '.jpg'
                    buf = io.BytesIO()
                    if save_fmt == 'PNG':
                        res.save(buf, format='PNG')
                    else:
                        res.save(buf, format='JPEG', quality=95)
                    t_clean = os.path.splitext(t_name)[0]
                    if t_clean.lower().endswith('.png'):
                        t_clean = t_clean[:-4]
                    zf.writestr(f"{base_name}/{t_clean}{save_ext}", buf.getvalue())
                    count += 1
                    progress.progress(count/total)
        st.session_state.zip_ready = True
        st.session_state.zip_data = zip_buf.getvalue()
        st.success("Batch Completato! Scritte rimosse automaticamente su tutte le copertine.")
    
    if st.session_state.get('zip_ready'):
        st.download_button("📥 SCARICA ZIP", st.session_state.zip_data, f"Mockups_{scelta}.zip", "application/zip")
