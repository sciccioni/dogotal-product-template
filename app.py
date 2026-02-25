import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, io, zipfile, json, base64, urllib.request, re

st.set_page_config(page_title="PhotoBook Mockup", layout="wide")

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- TEMPLATE MAPS ---
TEMPLATE_MAPS_FILE = "template_coordinates.json"

def load_template_maps():
    default = {
        "base_verticale_temi_app.jpg":        {"coords": (34.4, 9.1, 30.6, 80.4), "offset": 1},
        "base_orizzontale_temi_app.jpg":       {"coords": (18.9, 9.4, 61.9, 83.0), "offset": 1},
        "base_orizzontale_temi_app3.jpg":      {"coords": (18.7, 9.4, 62.2, 82.9), "offset": 1},
        "base_quadrata_temi_app.jpg":          {"coords": (27.7, 10.5, 44.7, 79.4), "offset": 1},
        "base_bottom_app.jpg":                 {"coords": (21.8, 4.7, 57.0, 91.7), "offset": 1},
        "15x22-crea la tua grafica.jpg":       {"coords": (33.1, 21.4, 33.9, 57.0), "offset": 2},
        "20x30-crea la tua grafica.jpg":       {"coords": (33.1, 21.4, 33.9, 57.0), "offset": 2},
        "Fotolibro-Temi-Verticali-temi-2.png": {"coords": (13.6, 4.0, 73.0, 92.0), "offset": 1},
        "Fotolibro-Temi-Verticali-temi-3.png": {"coords": (13.6, 4.0, 73.0, 92.0), "offset": 1},
    }
    if os.path.exists(TEMPLATE_MAPS_FILE):
        try:
            with open(TEMPLATE_MAPS_FILE) as f:
                return json.load(f)
        except:
            pass
    return default

def save_template_maps(maps):
    with open(TEMPLATE_MAPS_FILE, 'w') as f:
        json.dump(maps, f, indent=2)

TEMPLATE_MAPS = load_template_maps()

# --- CATEGORIE ---
def get_manual_cat(fn):
    fn = fn.lower()
    if any(x in fn for x in ["vertical","15x22","20x30","bottom","copertina_verticale"]): return "Verticali"
    if any(x in fn for x in ["orizzontal","20x15","27x20","32x24","40x30"]): return "Orizzontali"
    if any(x in fn for x in ["quadrat","20x20","30x30"]): return "Quadrati"
    return "Altro"

def get_folder_hash(p):
    if not os.path.exists(p): return 0
    return sum(os.path.getmtime(os.path.join(p,f)) for f in os.listdir(p))

# --- COMPOSITE ---
def find_book_region(tmpl_gray, bg_val):
    h, w = tmpl_gray.shape
    book_mask = tmpl_gray > (bg_val + 3)
    rows, cols = np.any(book_mask, axis=1), np.any(book_mask, axis=0)
    if not rows.any() or not cols.any(): return None
    by1, by2 = np.where(rows)[0][[0,-1]]
    bx1, bx2 = np.where(cols)[0][[0,-1]]
    mid_y = (by1+by2)//2
    row = tmpl_gray[mid_y]
    face_x1 = bx1
    for x in range(bx1, bx2-5):
        if np.all(row[x:x+5] >= 240): face_x1 = x; break
    return {'book_x1':int(bx1),'book_x2':int(bx2),'book_y1':int(by1),'book_y2':int(by2),'face_x1':int(face_x1)}

def composite_v3_fixed(tmpl_pil, cover_pil, template_name="", border_offset=None):
    has_alpha = False
    alpha_mask = None
    if tmpl_pil.mode in ('RGBA','LA') or (tmpl_pil.mode=='P' and 'transparency' in tmpl_pil.info) or template_name.lower().endswith('.png'):
        has_alpha = True
        tmpl_pil = tmpl_pil.convert('RGBA')
        alpha_mask = tmpl_pil.split()[3]
    tmpl_rgb = tmpl_pil.convert('RGB')
    h, w = tmpl_rgb.size[1], tmpl_rgb.size[0]
    if template_name in TEMPLATE_MAPS:
        d = TEMPLATE_MAPS[template_name]
        px,py,pw,ph = d["coords"]
        bo = border_offset if border_offset is not None else d.get("offset",1)
        x1,y1 = int(px*w/100)+bo, int(py*h/100)+bo
        tw,th = int(pw*w/100)-(bo*2), int(ph*h/100)-(bo*2)
        target_aspect = tw/th
        cw,ch = cover_pil.size
        if cw/ch > target_aspect:
            nw = int(ch*target_aspect)
            crop = ((cw-nw)//2,0,(cw-nw)//2+nw,ch)
        else:
            nh = int(cw/target_aspect)
            crop = (0,(ch-nh)//2,cw,(ch-nh)//2+nh)
        c_res = cover_pil.crop(crop).resize((tw,th),Image.LANCZOS)
        tmpl_l = np.array(tmpl_rgb.convert('L')).astype(np.float64)
        shadows = np.clip(tmpl_l[y1:y1+th,x1:x1+tw]/246.0,0,1.0)
        c_array = np.array(c_res.convert('RGB')).astype(np.float64)
        for i in range(3): c_array[:,:,i] *= shadows
        final_face = Image.fromarray(c_array.astype(np.uint8))
        tmpl_rgb.paste(final_face,(x1,y1))
        if has_alpha: tmpl_rgb.putalpha(alpha_mask)
        return tmpl_rgb
    tmpl_gray = np.array(tmpl_rgb.convert('L'))
    corners = [tmpl_gray[3,3],tmpl_gray[3,w-3],tmpl_gray[h-3,3],tmpl_gray[h-3,w-3]]
    bg_val = float(np.median(corners))
    region = find_book_region(tmpl_gray, bg_val)
    if region is None or (region['book_x2']-region['book_x1']>w*0.95):
        bx1,bx2,by1,by2 = int(w*0.2),w-int(w*0.2),int(h*0.1),h-int(h*0.1)
    else:
        bx1,bx2,by1,by2 = region['book_x1'],region['book_x2'],region['book_y1'],region['book_y2']
    bx1,bx2 = max(0,bx1-2),min(w-1,bx2+2)
    by1,by2 = max(0,by1-2),min(h-1,by2+2)
    tw,th = bx2-bx1+1,by2-by1+1
    c_res = cover_pil.resize((tw,th),Image.LANCZOS)
    c_arr = np.array(c_res.convert('RGB')).astype(np.float64)
    sh = np.clip(tmpl_gray[by1:by2+1,bx1:bx2+1]/246.0,0,1.0)
    for i in range(3): c_arr[:,:,i] *= sh
    final_face = Image.fromarray(c_arr.astype(np.uint8))
    tmpl_rgb.paste(final_face,(bx1,by1))
    if has_alpha: tmpl_rgb.putalpha(alpha_mask)
    return tmpl_rgb

# --- ANNO ENGINE ---
GPT_SIZE = 512

def trova_anno_gpt(img_pil, openai_api_key):
    orig_w, orig_h = img_pil.size
    img_small = img_pil.convert("RGB").resize((GPT_SIZE, GPT_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img_small.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        f"Questa immagine è esattamente {GPT_SIZE}x{GPT_SIZE} pixel. "
        "Trova SOLO il testo numerico dell'anno (es. 2024, 2025, 2026). "
        "NON includere il titolo principale (nome città/luogo). "
        "Rispondi SOLO con JSON: "
        '{"trovato":true,"x1":120,"y1":145,"x2":280,"y2":195} '
        f"coordinate pixel esatte nell'immagine {GPT_SIZE}x{GPT_SIZE}. "
        'Se non trovi: {"trovato":false}'
    )
    payload = json.dumps({
        "model": "gpt-4o-mini", "max_tokens": 80,
        "messages": [{"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}","detail":"high"}},
            {"type":"text","text":prompt}
        ]}]
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=payload,
        headers={"Content-Type":"application/json","Authorization":f"Bearer {openai_api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            raw = re.sub(r"```json|```","",data["choices"][0]["message"]["content"].strip()).strip()
            r = json.loads(raw)
            if not r.get("trovato"): return None
            sx, sy = orig_w/GPT_SIZE, orig_h/GPT_SIZE
            return int(r["x1"]*sx), int(r["y1"]*sy), int(r["x2"]*sx), int(r["y2"]*sy)
    except:
        return None

def elabora_testo_dinamico(img_pil, openai_api_key, azione="Rimuovi",
                            new_text_str="", font_file_bytes=None,
                            font_size=None, text_color_override=None):
    if azione == "Nessuna modifica":
        return img_pil
    if not openai_api_key:
        st.warning("⚠️ Inserisci la OpenAI API Key")
        return img_pil

    coords = trova_anno_gpt(img_pil, openai_api_key)
    if coords is None:
        st.info("ℹ️ Anno non trovato — immagine lasciata intatta.")
        return img_pil

    x1, y1, x2, y2 = coords
    img = img_pil.convert("RGB")
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(w-1,x2), min(h-1,y2)
    bbox_h = y2-y1
    bbox_w = x2-x1

    # Sfondo: campiona INTORNO al bbox (non dentro)
    pad = 25
    mask = np.zeros((h,w), dtype=bool)
    mask[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)] = True
    mask[y1:y2, x1:x2] = False
    bg_pixels = img_arr[mask]
    bg_color = tuple(int(v) for v in np.median(bg_pixels, axis=0)) if len(bg_pixels)>0 else (255,255,255)

    # Colore testo
    region = img_arr[y1:y2, x1:x2].reshape(-1,3)
    diffs = np.abs(region.astype(int) - np.array(bg_color)).sum(axis=1)
    text_pix = region[diffs > 60]
    if text_color_override:
        text_color = text_color_override
    elif len(text_pix) > 0:
        text_color = tuple(int(v) for v in np.median(text_pix, axis=0))
    else:
        text_color = (255,255,255) if sum(bg_color)/3 < 128 else (0,0,0)

    # Cancella
    draw = ImageDraw.Draw(img)
    padding = 12
    draw.rectangle([x1-padding, y1-padding, x2+padding, y2+padding], fill=bg_color)

    # Riscrivi
    if azione == "Sostituisci" and new_text_str and font_file_bytes:
        try:
            if font_size is None:
                fs = 8
                while fs < 500:
                    font = ImageFont.truetype(io.BytesIO(font_file_bytes), fs)
                    tb = font.getbbox(new_text_str)
                    if (tb[3]-tb[1]) >= bbox_h*0.85: break
                    fs += 1
            else:
                fs = font_size
            font = ImageFont.truetype(io.BytesIO(font_file_bytes), fs)
            tb = font.getbbox(new_text_str)
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
            draw_x = x2 - tw      # allineato a DESTRA
            draw_y = y1 + (bbox_h-th)//2
            draw.text((draw_x, draw_y), new_text_str, font=font, fill=text_color)
        except Exception as e:
            st.warning(f"Errore font: {e}")

    return img

# --- LIBRERIA ---
@st.cache_data
def get_lib(h_val):
    lib = {"Verticali":{},"Orizzontali":{},"Quadrati":{},"Altro":{}}
    if not os.path.exists("templates"): return lib
    for f in os.listdir("templates"):
        if f.lower().endswith(('.jpg','.png','.jpeg')):
            lib[get_manual_cat(f)][f] = Image.open(os.path.join("templates",f))
    return lib

libreria = get_lib(get_folder_hash("templates"))

# --- UI ---
menu = st.sidebar.radio("Menu", ["📚 Templates","🎯 Calibrazione","⚡ Produzione"])

if menu == "📚 Templates":
    st.subheader("📚 Libreria Templates")
    if st.button("🔄 RICARICA"):
        st.cache_data.clear(); st.rerun()
    ts = st.tabs(list(libreria.keys()))
    for i,c in enumerate(libreria.keys()):
        with ts[i]:
            cols = st.columns(4)
            for idx,(fn,img) in enumerate(libreria[c].items()):
                cols[idx%4].image(img, caption=fn, use_column_width=True)

elif menu == "🎯 Calibrazione":
    cat = st.selectbox("Categoria:", list(libreria.keys()))
    sel = st.selectbox("Template:", list(libreria[cat].keys()))
    if sel:
        t_img = libreria[cat][sel]
        d = TEMPLATE_MAPS.get(sel, {"coords":(20,10,60,80),"offset":1})
        if 'cal' not in st.session_state or st.session_state.get('cur') != sel:
            st.session_state.cal = d; st.session_state.cur = sel
        c = list(st.session_state.cal["coords"])
        col1,col2 = st.columns(2)
        c[0] = col1.number_input("X %",0.0,100.0,float(c[0]))
        c[1] = col2.number_input("Y %",0.0,100.0,float(c[1]))
        c[2] = col1.number_input("W %",0.0,100.0,float(c[2]))
        c[3] = col2.number_input("H %",0.0,100.0,float(c[3]))
        st.session_state.cal["offset"] = st.slider("Offset",0,20,int(st.session_state.cal["offset"]))
        p_img = t_img.copy().convert('RGB')
        draw = ImageDraw.Draw(p_img)
        ww,hh = p_img.size
        draw.rectangle([int(c[0]*ww/100),int(c[1]*hh/100),int((c[0]+c[2])*ww/100),int((c[1]+c[3])*hh/100)],outline="red",width=5)
        st.image(p_img, use_column_width=True)
        if st.button("💾 SALVA"):
            TEMPLATE_MAPS[sel] = st.session_state.cal
            save_template_maps(TEMPLATE_MAPS)
            st.success("Salvate!")

elif menu == "⚡ Produzione":
    scelta = st.radio("Formato:", ["Verticali","Orizzontali","Quadrati"], horizontal=True)

    with st.expander("🤖 Sostituzione Anno", expanded=True):
        openai_key = st.text_input("🔑 OpenAI API Key", type="password")

        modalita_testo = st.radio(
            "Azione:",
            ("Nessuna modifica","Solo Rimuovi Anno","Rimuovi e Sostituisci Anno"),
            index=0, horizontal=True
        )

        font_bytes = None
        new_text_input = ""
        font_size_input = None
        color_override = None
        usa_colore_auto = True

        if modalita_testo == "Rimuovi e Sostituisci Anno":
            col1,col2,col3 = st.columns(3)
            new_text_input = col1.text_input("Testo sostitutivo", value="2026")
            font_file = col2.file_uploader("Font (.ttf/.otf)", type=['ttf','otf'])
            if font_file:
                font_bytes = font_file.read()
                col2.success("✅ Font caricato!")
            else:
                col2.warning("Carica un font .ttf")

            usa_auto = col3.checkbox("🔁 Dimensione automatica (matcha originale)", value=True)
            if not usa_auto:
                font_size_input = col3.number_input("Dimensione font (px)", value=80, min_value=8, max_value=500)

            usa_colore_auto = col1.checkbox("🎨 Colore automatico", value=True)
            if not usa_colore_auto:
                hex_c = col1.color_picker("Colore testo", "#FFD700")
                hex_c = hex_c.lstrip('#')
                color_override = tuple(int(hex_c[i:i+2],16) for i in (0,2,4))

    st.divider()

    def processa_img(img_pil):
        if modalita_testo == "Nessuna modifica":
            return img_pil
        azione = "Sostituisci" if modalita_testo == "Rimuovi e Sostituisci Anno" else "Rimuovi"
        return elabora_testo_dinamico(
            img_pil            = img_pil,
            openai_api_key     = openai_key,
            azione             = azione,
            new_text_str       = new_text_input,
            font_file_bytes    = font_bytes,
            font_size          = font_size_input,
            text_color_override= color_override if not usa_colore_auto else None
        )

    # Anteprima
    up = st.file_uploader("Carica design singolo (Anteprima)", type=['jpg','png'], key='preview')
    if up and libreria[scelta]:
        d_img = Image.open(up)
        d_img = processa_img(d_img)
        st.image(d_img, caption="Design Processato", width=300)
        cols = st.columns(4)
        for i,(t_name,t_img) in enumerate(libreria[scelta].items()):
            with cols[i%4]:
                res = composite_v3_fixed(t_img, d_img, t_name)
                st.image(res, caption=t_name, use_column_width=True)

    st.divider()

    # Batch
    batch = st.file_uploader("Batch Produzione", accept_multiple_files=True)
    if st.button("🚀 GENERA TUTTI (BATCH)") and batch and libreria[scelta]:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf,"a",zipfile.ZIP_DEFLATED) as zf:
            progress = st.progress(0)
            total = len(batch)*len(libreria[scelta])
            count = 0
            for b_file in batch:
                b_img = Image.open(b_file)
                b_img = processa_img(b_img)
                base_name = os.path.splitext(b_file.name)[0]
                for t_name,t_img in libreria[scelta].items():
                    res = composite_v3_fixed(t_img, b_img, t_name)
                    is_png = t_name.lower().endswith('.png') or res.mode=='RGBA'
                    buf = io.BytesIO()
                    if is_png: res.save(buf,format='PNG')
                    else: res.save(buf,format='JPEG',quality=95)
                    ext = '.png' if is_png else '.jpg'
                    t_clean = os.path.splitext(t_name)[0]
                    zf.writestr(f"{base_name}/{t_clean}{ext}", buf.getvalue())
                    count += 1
                    progress.progress(count/total)
        st.session_state.zip_ready = True
        st.session_state.zip_data = zip_buf.getvalue()
        st.success("✅ Batch Completato!")

    if st.session_state.get('zip_ready'):
        st.download_button("📥 SCARICA ZIP", st.session_state.zip_data,
                           f"Mockups_{scelta}.zip","application/zip")
