"""Office purchase receipt scanner: capture -> OCR -> parse -> review -> save to CSV/XLSX.
Free and fully offline (RapidOCR / PaddleOCR models via ONNX).
Run:  streamlit run app.py
"""
import hashlib
import io
import re
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dateparser
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

RECORDS_DIR = Path("records")
RECORDS_DIR.mkdir(exist_ok=True)
COLUMNS = ["Date", "Item", "Price"]

# Lines that are not purchased items
SKIP = re.compile(
    r"\b(sub\s*-?total|total|vat|tax|change|cash|tender|balance|paid|amount|"
    r"m-?pesa|visa|card|tel|phone|pin|receipt|invoice|served|cashier|thank|"
    r"welcome|discount|rounding|till|paybill|account|pos|ref)\b",
    re.I,
)
PRICE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?![\d])")
TOTAL = re.compile(r"\b(grand\s*total|total\s*(?:due|amount)?|amount\s*due)\b", re.I)
DATE_PATTERNS = [
    r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
    r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b",
    r"\b\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s,.-]*\d{2,4}\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b",
]

@st.cache_resource
def get_engine():
    return RapidOCR()

def preprocess(img: Image.Image) -> np.ndarray:
    """Fix phone rotation, upscale small images, boost local contrast."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    if max(h, w) < 1400:
        s = 1400 / max(h, w)
        arr = cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > 2600:
        s = 2600 / max(h, w)
        arr = cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def ocr_lines(arr: np.ndarray):
    """Run OCR and group word boxes into visual rows."""
    result, _ = get_engine()(arr)
    if not result:
        return []
    items = []
    for box, text, conf in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append(
            dict(x=min(xs), yc=sum(ys) / 4, h=max(ys) - min(ys), text=text.strip())
        )
    items.sort(key=lambda d: d["yc"])
    med_h = float(np.median([d["h"] for d in items])) or 20
    rows, cur = [], [items[0]]
    for d in items[1:]:
        ref = sum(c["yc"] for c in cur) / len(cur)
        if abs(d["yc"] - ref) <= 0.6 * med_h:
            cur.append(d)
        else:
            rows.append(cur)
            cur = [d]
    rows.append(cur)
    out = []
    for r in rows:
        r.sort(key=lambda d: d["x"])
        out.append((" ".join(d["text"] for d in r), [(d["x"], d["text"]) for d in r]))
    return out

def find_date(lines):
    text = "\n".join(t for t, _ in lines)
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            try:
                d = dateparser.parse(m.group(0), dayfirst=True, fuzzy=True).date()
                if date(2000, 1, 1) <= d <= date.today():
                    return d
            except (ValueError, OverflowError):
                continue
    return None

MONEY = re.compile(r"(?<![\d.,])\d+(?:,\d{3})*\.\d{2}(?![\d.])")
INT_END = re.compile(r"^(.*[A-Za-z]{3}.*?)\s+(\d{2,6})$")
UNIT = re.compile(r"\d+(?:\.\d+)?\s*(?:pcs?|kgs?|g|ltrs?|l|units?|ea|x)\b", re.I)
HEADER = re.compile(r"\bitem\b.*\b(total|price|amount|each|qty)\b", re.I)
END = re.compile(
    r"^\W*(sub\s*-?total|grand\s*total|total|amount\s*due|cash|m-?pesa|change|paid|tender|balance)\b",
    re.I,
)
_CONFUSE = str.maketrans("01584", "OISBA")

def norm(t):
    return t.upper().translate(_CONFUSE)

def to_number(text):
    return float(text.replace(",", ""))

def fix_ocr(name):
    name = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "O", name)
    return re.sub(
        r"\b([0-9O]{2,4})(ML|G|KG|L)\b",
        lambda m: m.group(1).replace("O", "0") + m.group(2),
        name,
        flags=re.I,
    )

def clean_name(name):
    name = re.sub(r"^\d{1,3}\s*[xX*]\s+", "", name.strip())
    name = re.sub(r"\s+[A-E]$", "", name.strip())
    return fix_ocr(name.strip(" .:-*@"))

def parse_receipt(lines):
    """Return (vendor, [{Item, Price}], receipt_total)."""
    texts = [t for t, _ in lines]
    vendor = next((t for t in texts[:4] if re.search(r"[A-Za-z]{3,}", t)), "")

    start = next((i + 1 for i, t in enumerate(texts) if HEADER.search(t)), None)
    end = next(
        (i for i, t in enumerate(texts) if i >= (start or 0) and END.search(norm(t))),
        len(texts),
    )
    total = None
    if end < len(texts):
        m = MONEY.findall(texts[end])
        if m:
            total = to_number(m[-1])
        else:
            n = re.search(r"(\d[\d,]*)\s*$", texts[end])
            total = to_number(n.group(1)) if n else None
    block = texts[start if start is not None else 0 : end]

    rows, pending = [], None
    for t in block:
        t = t.strip()
        if not t or (start is None and SKIP.search(t)):
            continue
        prices = MONEY.findall(t)
        letters = len(re.findall(r"[A-Za-z]", t))
        is_detail = bool(UNIT.search(t)) or (prices and letters < 4)
        if is_detail and pending and prices:
            rows.append(dict(Item=clean_name(pending), Price=to_number(prices[-1])))
            pending = None
        elif prices and letters >= 3:
            name = t[: t.rfind(prices[-1])]
            rows.append(dict(Item=clean_name(name), Price=to_number(prices[-1])))
            pending = None
        elif letters >= 3:
            m = INT_END.match(t)
            if m:
                rows.append(dict(Item=clean_name(m.group(1)), Price=float(m.group(2))))
                pending = None
            else:
                pending = t if pending is None else f"{pending} {t}"
    rows = [
        r
        for r in rows
        if len(re.findall(r"[A-Za-z]", r["Item"])) >= 2 and not END.search(norm(r["Item"]))
    ]
    return vendor, rows, total

def save(df: pd.DataFrame, fmt: str) -> Path:
    path = RECORDS_DIR / f"purchases.{fmt}"
    if path.exists():
        old = pd.read_csv(path) if fmt == "csv" else pd.read_excel(path)
        df = pd.concat([old, df], ignore_index=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)
    return path

@st.cache_data(show_spinner=False)
def process(data: bytes) -> dict:
    arr = preprocess(Image.open(io.BytesIO(data)))
    lines = ocr_lines(arr)
    vendor, rows, total = parse_receipt(lines) if lines else ("", [], None)
    return dict(
        vendor=vendor,
        rows=rows,
        total=total,
        date=find_date(lines) if lines else None,
        raw="\n".join(t for t, _ in lines),
    )

# --- UI SETUP & CUSTOM STYLING ---
st.set_page_config(page_title="Snap & Save", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -15px;
    }
    .sub-header {
        text-align: center;
        color: #555;
        margin-bottom: 30px;
        font-size: 0.95rem;
    }
    /* Expander card UI styling */
    [data-testid="stExpander"] {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0,0,0,0.05);
        background-color: #ffffff;
    }
    /* Primary button glow */
    [data-testid="baseButton-primary"] {
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 14px 0 rgba(78, 67, 118, 0.25);
        transition: transform 0.2s;
    }
    [data-testid="baseButton-primary"]:active {
        transform: scale(0.98);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Snap & Save</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Scan office receipts instantly to CSV or Excel</div>", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("cam", {})
ss.setdefault("cam_key", 0)  # Used to reset the camera for batch scanning
ss.setdefault("saved", None)
ss.setdefault("fmt", "xlsx")

# Top Metrics Container - 2x2 grid for mobile friendliness
summary = st.empty()

# Input Tabs
tab_cam, tab_up = st.tabs(["Take Photo", "Upload Batch"])

with tab_cam:
    st.info("Capture a clear, well-lit photo of the receipt.")
    
    # By tying the key to ss.cam_key, the widget resets immediately after processing a shot
    shot = st.camera_input("Take a photo", label_visibility="collapsed", key=f"camera_input_{ss.cam_key}")
    
    if shot:
        b = shot.getvalue()
        file_hash = hashlib.md5(b).hexdigest()
        
        if file_hash not in ss.cam:
            ss.cam[file_hash] = b
            ss.cam_key += 1  # Increment to reset the camera UI
            st.rerun()       # Force UI refresh to show clean camera instantly
            
    if ss.cam:
        st.success(f"{len(ss.cam)} photo(s) queued for processing in this batch.")

with tab_up:
    uploads = st.file_uploader(
        "Or upload images from your gallery",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

# Build the batch
batch = {}
for f in uploads or []:
    b = f.getvalue()
    batch.setdefault(hashlib.md5(b).hexdigest(), (f.name, b))
for n, (h, b) in enumerate(ss.cam.items(), 1):
    batch.setdefault(h, (f"Camera_Capture_{n}", b))

if not batch:
    st.stop()

st.divider()

# Processing state
results = {}
progress_bar = st.progress(0.0)
status_text = st.empty()

for i, (h, (name, b)) in enumerate(batch.items(), 1):
    status_text.caption(f"Reading receipt {i} of {len(batch)}...")
    results[h] = process(b)
    progress_bar.progress(i / len(batch))

progress_bar.empty()
status_text.empty()

final, n_checked, n_warn, grand = [], 0, 0, 0.0

st.markdown("### Review Items")

for h, (name, b) in batch.items():
    r = results[h]
    df = pd.DataFrame(r["rows"], columns=["Item", "Price"])
    df.insert(0, "Date", r["date"] or date.today())
    
    items_sum = float(df["Price"].sum()) if len(df) else 0.0
    ok = r["total"] is not None and abs(r["total"] - items_sum) < 0.5
    
    status = "[OK]" if ok else "[CHECK]"
    label = f"{status} {name} | {len(df)} items | Total: ${items_sum:,.2f}"
    
    with st.expander(label, expanded=not ok):
        include = st.checkbox(f"Include {name} in final save", value=True, key=f"inc_{h}")
        
        # Display image first on mobile (stacked)
        st.image(Image.open(io.BytesIO(b)), use_container_width=True)
        
        if not r["rows"]:
            st.error("No items detected. Try better lighting or add rows below.")
        if r["date"] is None:
            st.info("No date found. Defaults to today.")
            
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"ed_{h}",
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Price": st.column_config.NumberColumn("Amount", format="%.2f", step=0.5),
            },
        )
        
        e_sum = float(edited["Price"].sum()) if len(edited) else 0.0
        if r["total"] is not None:
            if abs(r["total"] - e_sum) < 0.5:
                st.success(f"Receipt total: {r['total']:,.2f} (Matches items)")
            else:
                st.warning(f"Receipt total: {r['total']:,.2f} | Items total: {e_sum:,.2f} (Mismatch)")
                
        with st.expander("Show Raw Text (Debug)"):
            st.code(r["raw"])

    if not ok:
        n_warn += 1
    if include:
        keep = edited.dropna(subset=["Item", "Price"])
        final.append(keep)
        grand += float(keep["Price"].sum())

# Populate Top Summary metrics (2x2 grid)
all_rows = pd.concat(final, ignore_index=True) if final else pd.DataFrame(columns=COLUMNS)

with summary.container():
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    c1.metric("Receipts Processed", len(batch))
    c2.metric("Total Items", len(all_rows))
    c3.metric("Grand Total", f"${grand:,.2f}")
    c4.metric("Needs Review", n_warn)

st.divider()

# Save Action (Full-width button)
if st.button(f"Save {len(all_rows)} Item(s)", type="primary", disabled=all_rows.empty, use_container_width=True):
    ss.saved = save(all_rows, ss.fmt)
    st.success(f"Saved successfully to {ss.saved}")

if ss.saved and Path(ss.saved).exists():
    st.download_button(
        label=f"Download {Path(ss.saved).name}",
        data=Path(ss.saved).read_bytes(),
        file_name=Path(ss.saved).name,
        use_container_width=True
    )

# Advanced Settings
with st.expander("Advanced Settings"):
    ss.fmt = st.radio("Export Format", ["xlsx", "csv"], index=0, horizontal=True)
    if ss.cam:
        if st.button("Clear Camera Session"):
            ss.cam = {}
            ss.cam_key = 0
            st.rerun()