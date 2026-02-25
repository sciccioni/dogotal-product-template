import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import io
import zipfile
import json
import scipy.ndimage as ndi  # <-- IL NUOVO CERVELLO MATEMATICO

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="PhotoBook Mockup - V9 FULL AUTO", layout="wide")

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


# --- NUOVO CERVELLO: TROVA ANNO IN AUTOMATICO ---
def elabora_testo_dinamico(img_pil, sample_x, sample_y, azione="Rimuovi", new_text_str="", font_obj=None, x_offset=0, y_offset=0, padding=10):
    """
    Trova dinamicamente il testo in basso a destra (l'anno), cancella solo quello e
    lo sostituisce ancorandolo a DESTRA per non farlo sbordare. Zero coordinate manuali.
    """
    img = img_pil.convert("RGB")
    img_arr = np.array(img)
    h, w, _ = img_arr.shape
    
    # 1. Trova il colore di sfondo
    bg_color = img_arr[sample_y, sample_x]
    
    # 2. Maschera tutto ciò che non è sfondo
    diff = np.abs(img_arr.astype(int) - bg_color.astype(int))
    mask = np.sum(diff, axis=2) > 40
    
    if not np.any(mask): 
        return img # Immagine vuota o tutta tinta unita
        
    # 3. DILATAZIONE: Usa scipy per "fondere" le lettere vicine e formare parole/blocchi solidi
    # 15 iterazioni chiudono i gap tra i numeri 2, 0, 2, 5.
    dilated = ndi.binary_dilation(mask, iterations=15)
    
    # 4. Etichetta i blocchi distinti (es. ROMA = blocco 1, 2025 = blocco 2)
    labels, num_features = ndi.label(dilated)
    
    if num_features == 0: 
        return img
        
    # 5. TROVA L'ANNO: Euristica intelligente. L'anno è sempre il blocco più in basso e a destra.
    # Calcoliamo X_MAX + Y_MAX per ogni blocco. Quello col punteggio più alto vince.
    best_bbox = None
    max_score = -1
    best_blob_mask = None
    
    for i in range(1, num_features + 1):
        blob_mask = (labels == i)
        rows = np.any(blob_mask, axis=1)
        cols = np.any(blob_mask, axis=0)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        score = xmax + ymax  # Il vero trucco è qui
        
        if score > max_score:
            max_score = score
            best_bbox = (xmin, ymin, xmax, ymax)
            best_blob_mask = blob_mask
            
    if not best_bbox: return img
    
    xmin, ymin, xmax, ymax = best_bbox
    
    # 6. Colore del testo: Prende i pixel originali (non dilatati) dentro a questo blocco
    text_pixels = img_arr[mask & best_blob_mask]
    if len(text_pixels) > 0:
        avg_color = np.median(text_pixels, axis=0).astype(int)
    else:
        avg_color = np.array([255, 255, 255])
        
    # 7. CANCELLA IL VECCHIO TESTO
    draw = ImageDraw.Draw(img)
    erase_x1 = max(0, xmin - padding)
    erase_y1 = max(0, ymin - padding)
    erase_x2 = min(w, xmax + padding)
    erase_y2 = min(h, ymax + padding)
    
    # Toppa del colore di sfondo esatto
    draw.rectangle([erase_x1, erase_y1, erase_x2, erase_y2], fill=tuple(bg_color))
    
    # 8. SCRIVI IL NUOVO TESTO (SE RICHIESTO)
    if azione == "Sostituisci" and new_text_str and font_obj:
        # Prendi le misure reali del nuovo testo
        bbox = font_obj.getbbox(new_text_str)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # ALLINEAMENTO A DESTRA: Il punto di partenza X è il vecchio margine destro MENO la larghezza del testo.
        # Così non supererà mai l'ingombro originale del 2025.
        draw_x = xmax - text_width + x_offset
        
        # ALLINEAMENTO VERTICALE: Lo posizioniamo in modo che poggi sulla stessa "linea di base" del vecchio anno.
        draw_y = ymax - text_height + y_offset
        
        draw.text((draw_x, draw_y), new_text_str, font=font_obj, fill=tuple(avg_color))
    
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
    
    with st.expander("🤖 Sostituzione Anno AUTOMATICA", expanded=True):
        st.write("Questa modalità scansiona il design, trova in automatico il blocco di testo più in basso a destra (l'anno) e lo sostituisce allineandolo a destra per non farlo sbordare. Addio riquadri manuali!")
        
        modalita_testo = st.radio(
            "Scegli l'azione da eseguire:",
            ("Nessuna modifica", "Solo Rimuovi Anno", "Rimuovi e Sostituisci Anno"),
            index=0,
            horizontal=True
        )
        
        if modalita_testo != "Nessuna modifica":
            # Teniamo solo il pallino del colore di sfondo (sicuro, in alto a sinistra)
            col_cen1, col_cen2 = st.columns(2)
            samp_x = col_cen1.number_input("X Colore Sfondo (Alto a sx)", 0, 1000, 10)
            samp_y = col_cen2.number_input("Y Colore Sfondo (Alto a sx)", 0, 1000, 10)
            st.divider()

        if modalita_testo == "Rimuovi e Sostituisci Anno":
            col_txt1, col_txt2 = st.columns(2)
            new_text_input = col_txt1.text_input("Nuovo Anno", value="2026")
            font_size_input = col_txt2.number_input("Dimensione Font", value=80, min_value=10)
            font_file = col_txt1.file_uploader("Carica file Font (.ttf o .otf)", type=['ttf', 'otf'])
            
            st.caption("Correzioni fini (Usa solo se il font risulta storto):")
            x_offset_val = col_txt2.slider("X Offset", -100, 100, 0, help="Sposta il testo orizzontalmente")
            y_offset_val = col_txt2.slider("Y Offset", -100, 100, 0, help="Sposta il testo verticalmente")

            loaded_font = None
            if font_file:
                try:
                    loaded_font = ImageFont.truetype(io.BytesIO(font_file.read()), font_size_input)
                    col_txt1.success("Font caricato!")
                except:
                    col_txt1.error("Errore font.")
            else:
                col_txt1.warning("Carica un font!")
            st.divider()

    up = st.file_uploader("Carica design singolo (Anteprima)", type=['jpg', 'png'], key='preview')
    if up and libreria[scelta]:
        d_img = Image.open(up)
        
        if modalita_testo != "Nessuna modifica":
            azione_passata = "Sostituisci" if modalita_testo == "Rimuovi e Sostituisci Anno" else "Rimuovi"
            font_to_pass = loaded_font if modalita_testo == "Rimuovi e Sostituisci Anno" else None
            
            d_img = elabora_testo_dinamico(
                d_img, samp_x, samp_y, 
                azione=azione_passata, 
                new_text_str=new_text_input if modalita_testo == "Rimuovi e Sostituisci Anno" else "", 
                font_obj=font_to_pass, 
                x_offset=x_offset_val if modalita_testo == "Rimuovi e Sostituisci Anno" else 0, 
                y_offset=y_offset_val if modalita_testo == "Rimuovi e Sostituisci Anno" else 0
            )
            st.success("Testo isolato e modificato in automatico!")
            st.image(d_img, caption="Design Processato", width=300)

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
                
                if modalita_testo != "Nessuna modifica":
                    azione_passata = "Sostituisci" if modalita_testo == "Rimuovi e Sostituisci Anno" else "Rimuovi"
                    font_to_pass = loaded_font if modalita_testo == "Rimuovi e Sostituisci Anno" else None
                    
                    b_img = elabora_testo_dinamico(
                        b_img, samp_x, samp_y, 
                        azione=azione_passata, 
                        new_text_str=new_text_input if modalita_testo == "Rimuovi e Sostituisci Anno" else "", 
                        font_obj=font_to_pass, 
                        x_offset=x_offset_val if modalita_testo == "Rimuovi e Sostituisci Anno" else 0, 
                        y_offset=y_offset_val if modalita_testo == "Rimuovi e Sostituisci Anno" else 0
                    )

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
        st.success("Batch Completato! Ogni anno è stato rilevato e allineato dinamicamente.")
    
    if st.session_state.get('zip_ready'):
        st.download_button("📥 SCARICA ZIP", st.session_state.zip_data, f"Mockups_{scelta}.zip", "application/zip")
