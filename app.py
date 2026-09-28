"""Office purchase receipt scanner: capture -> OCR -> parse -> review -> save to CSV/XLSX.
Free and fully offline (Tesseract OCR).
Run:  streamlit run app.py
"""
import base64
import hashlib
import io
import logging
import re
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from dateutil import parser as dateparser
from PIL import Image, ImageOps
from pytesseract import Output

LOGGER = logging.getLogger(__name__)


REAR_CAMERA_HTML = """
<div class="camera-shell">
  <div class="camera-view">
    <video id="camera-video" autoplay playsinline muted></video>
    <div id="camera-guide" aria-hidden="true"></div>
    <p id="quality-indicator" class="quality waiting">Checking light and focus…</p>
    <p id="camera-status" role="status">Starting the rear camera…</p>
  </div>
  <div class="camera-actions">
    <button id="capture-button" class="capture-button" type="button" disabled>
      <span aria-hidden="true">●</span> Capture receipt
    </button>
    <button id="switch-button" class="switch-button" type="button" disabled>
      Switch camera
    </button>
  </div>
  <canvas id="camera-canvas" hidden></canvas>
</div>
"""

REAR_CAMERA_CSS = """
:host {
  display: block;
  width: 100%;
  color: var(--st-text-color);
  font-family: var(--st-font);
}
.camera-shell { width: 100%; }
.camera-view {
  position: relative;
  width: 100%;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--st-text-color) 18%, transparent);
  border-radius: var(--st-border-radius-lg, 14px);
  background: #101114;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
}
#camera-video {
  display: block;
  width: 100%;
  height: min(68svh, 720px);
  min-height: 420px;
  object-fit: contain;
  background: #101114;
}
#camera-guide {
  position: absolute;
  inset: 7% 8%;
  pointer-events: none;
  border: 2px solid rgba(255, 255, 255, 0.78);
  border-radius: 12px;
  box-shadow: 0 0 0 999px rgba(0, 0, 0, 0.08);
}
#camera-status {
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 10px;
  margin: 0;
  padding: 8px 10px;
  border-radius: 8px;
  color: #fff;
  background: rgba(0, 0, 0, 0.62);
  font-size: 0.88rem;
  text-align: center;
}
.quality {
  position: absolute;
  top: 10px;
  left: 50%;
  z-index: 2;
  transform: translateX(-50%);
  width: max-content;
  max-width: calc(100% - 24px);
  margin: 0;
  padding: 7px 11px;
  border-radius: 999px;
  color: #fff;
  background: rgba(0, 0, 0, 0.68);
  font-size: 0.84rem;
  font-weight: 650;
  text-align: center;
}
.quality.good { background: rgba(24, 122, 72, 0.88); }
.quality.warning { background: rgba(180, 83, 9, 0.92); }
.camera-actions {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
  gap: 10px;
  margin-top: 12px;
}
.camera-actions button {
  min-height: 48px;
  padding: 10px 14px;
  border-radius: var(--st-border-radius, 8px);
  font: inherit;
  font-weight: 650;
  cursor: pointer;
}
.camera-actions button:disabled { cursor: not-allowed; opacity: 0.55; }
.capture-button {
  border: 1px solid var(--st-primary-color);
  color: var(--st-primary-button-text-color, #fff);
  background: var(--st-primary-color);
}
.capture-button span { margin-right: 6px; }
.switch-button {
  border: 1px solid color-mix(in srgb, var(--st-text-color) 28%, transparent);
  color: var(--st-text-color);
  background: var(--st-secondary-background-color);
}
@media (max-width: 640px) {
  #camera-video { height: 66svh; min-height: 430px; }
  #camera-guide { inset: 5% 5%; }
  .camera-actions { grid-template-columns: 1fr; }
  .camera-actions button { width: 100%; min-height: 52px; }
}
"""

REAR_CAMERA_JS = """
const mountedCameras = new WeakMap();

export default function cameraComponent(component) {
  const { parentElement, setTriggerValue } = component;
  const video = parentElement.querySelector("#camera-video");
  const canvas = parentElement.querySelector("#camera-canvas");
  const captureButton = parentElement.querySelector("#capture-button");
  const switchButton = parentElement.querySelector("#switch-button");
  const status = parentElement.querySelector("#camera-status");
  const quality = parentElement.querySelector("#quality-indicator");
  if (!video || !canvas || !captureButton || !switchButton || !status || !quality) return;

  let camera = mountedCameras.get(parentElement);
  if (!camera) {
    camera = {
      stream: null,
      devices: [],
      activeIndex: 0,
      stopped: false,
      qualityTimer: null,
      brightness: 0,
      sharpness: 0,
    };
    mountedCameras.set(parentElement, camera);
  }

  const stopStream = () => {
    if (camera.qualityTimer) {
      clearInterval(camera.qualityTimer);
      camera.qualityTimer = null;
    }
    if (camera.stream) {
      camera.stream.getTracks().forEach((track) => track.stop());
      camera.stream = null;
    }
  };

  const updateQuality = () => {
    if (!video.videoWidth || !video.videoHeight) return;
    canvas.width = 160;
    canvas.height = 120;
    const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let light = 0;
    let edges = 0;
    let previous = 0;
    const count = pixels.length / 4;
    for (let i = 0; i < pixels.length; i += 4) {
      const value = 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      light += value;
      if (i > 0) edges += Math.abs(value - previous);
      previous = value;
    }
    camera.brightness = light / count;
    camera.sharpness = edges / Math.max(1, count - 1);

    if (camera.brightness < 58) {
      quality.className = "quality warning";
      quality.textContent = "Too dark — add light or move closer";
    } else if (camera.sharpness < 5.5) {
      quality.className = "quality warning";
      quality.textContent = "Hold steady and tap the receipt to focus";
    } else {
      quality.className = "quality good";
      quality.textContent = "Ready — lighting and focus look good";
    }
  };

  const listCameras = async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();
    camera.devices = devices.filter((device) => device.kind === "videoinput");
    const rearIndex = camera.devices.findIndex((device) =>
      /back|rear|environment/i.test(device.label)
    );
    if (rearIndex >= 0) camera.activeIndex = rearIndex;
    switchButton.disabled = camera.devices.length < 2;
  };

  const startCamera = async (deviceId = null) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      status.textContent = "Camera access is not supported in this browser. Use Upload batch instead.";
      return;
    }
    stopStream();
    captureButton.disabled = true;
    status.hidden = false;
    status.textContent = "Starting the rear camera…";

    const videoConstraints = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
      : { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } };

    try {
      camera.stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });
      video.srcObject = camera.stream;
      await video.play();
      await listCameras();
      captureButton.disabled = false;
      status.hidden = true;
      updateQuality();
      camera.qualityTimer = setInterval(updateQuality, 650);
    } catch (error) {
      status.hidden = false;
      status.textContent = error?.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access or use Upload batch."
        : "Could not open the camera. Use Upload batch or try another browser.";
    }
  };

  captureButton.onclick = () => {
    if (!video.videoWidth || !video.videoHeight) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { alpha: false });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    status.hidden = false;
    status.textContent = "Photo captured. Preparing receipt…";
    setTriggerValue("captured", {
      dataUrl: canvas.toDataURL("image/jpeg", 0.94),
      brightness: camera.brightness,
      sharpness: camera.sharpness,
    });
  };

  switchButton.onclick = async () => {
    if (camera.devices.length < 2) return;
    camera.activeIndex = (camera.activeIndex + 1) % camera.devices.length;
    await startCamera(camera.devices[camera.activeIndex].deviceId);
  };

  if (!camera.stream && !camera.stopped) startCamera();

  return () => {
    camera.stopped = true;
    stopStream();
    mountedCameras.delete(parentElement);
  };
}
"""

REAR_CAMERA = st.components.v2.component(
    "rear_receipt_camera",
    html=REAR_CAMERA_HTML,
    css=REAR_CAMERA_CSS,
    js=REAR_CAMERA_JS,
)

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

def _order_corners(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral corners as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_diff = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_diff)]
    ordered[3] = points[np.argmax(coordinate_diff)]
    return ordered


def _perspective_crop(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    corners = _order_corners(corners.astype(np.float32))
    top_left, top_right, bottom_right, bottom_left = corners
    width = int(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    height = int(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if width < 200 or height < 300:
        return image
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _find_receipt_corners(image: np.ndarray) -> np.ndarray | None:
    """Find the largest plausible four-sided receipt boundary."""
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = small.shape[0] * small.shape[1] * 0.12
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        if cv2.contourArea(contour) < minimum_area:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            return polygon.reshape(4, 2).astype(np.float32) / scale
    return None


def _encode_preview(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not success:
        raise ValueError("Could not create the scanned receipt preview")
    return encoded.tobytes()


def preprocess(img: Image.Image) -> tuple[list[Image.Image], bytes, dict]:
    """Detect, straighten, trim, and enhance a receipt for OCR."""
    rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
    original_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    quality = {
        "brightness": round(float(np.mean(original_gray)), 1),
        "contrast": round(float(np.std(original_gray)), 1),
        "sharpness": round(float(cv2.Laplacian(original_gray, cv2.CV_64F).var()), 1),
        "cropped": False,
    }

    corners = _find_receipt_corners(rgb)
    scanned = _perspective_crop(rgb, corners) if corners is not None else rgb
    quality["cropped"] = corners is not None

    height, width = scanned.shape[:2]
    if max(height, width) < 1600:
        scale = 1600 / max(height, width)
        scanned = cv2.resize(
            scanned, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    elif max(height, width) > 2800:
        scale = 2800 / max(height, width)
        scanned = cv2.resize(
            scanned, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(scanned, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    enhanced = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        13,
    )

    warnings = []
    if quality["brightness"] < 62:
        warnings.append("The original photo was dark; add more light for better accuracy.")
    if quality["sharpness"] < 55:
        warnings.append("The photo may be blurred; hold the phone steady and refocus.")
    if quality["contrast"] < 24:
        warnings.append("The receipt has low contrast; avoid glare and shadows.")
    if not quality["cropped"]:
        warnings.append("Receipt edges were not confidently detected; fill more of the guide.")
    quality["warnings"] = warnings

    variants = [Image.fromarray(enhanced), Image.fromarray(binary)]
    return variants, _encode_preview(scanned), quality


def ocr_lines(img: Image.Image, page_mode: int = 6):
    """Run Tesseract and return grouped rows plus a confidence score."""
    data = pytesseract.image_to_data(
        img,
        lang="eng",
        config=f"--oem 1 --psm {page_mode} -c preserve_interword_spaces=1",
        output_type=Output.DICT,
    )
    grouped = {}
    confidences = []
    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1
        if not text or confidence < 15:
            continue
        confidences.append(confidence)
        key = (
            data["page_num"][i],
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )
        grouped.setdefault(key, []).append(
            {
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "text": text,
            }
        )

    rows = sorted(
        grouped.values(),
        key=lambda row: (min(word["y"] for word in row), min(word["x"] for word in row)),
    )
    out = []
    for row in rows:
        row.sort(key=lambda word: word["x"])
        out.append(
            (
                " ".join(word["text"] for word in row),
                [(word["x"], word["text"]) for word in row],
            )
        )
    money_hits = sum(bool(MONEY.search(text)) for text, _ in out)
    score = sum(confidences) + 18 * len(confidences) + 60 * money_hits
    return out, score

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

@st.cache_data(show_spinner=False, max_entries=128)
def process(data: bytes) -> dict:
    variants, preview, quality = preprocess(Image.open(io.BytesIO(data)))
    attempts = [
        ocr_lines(variants[0], 6),
        ocr_lines(variants[1], 6),
        ocr_lines(variants[0], 4),
    ]
    candidates = []
    for lines, ocr_score in attempts:
        vendor, rows, total = parse_receipt(lines) if lines else ("", [], None)
        items_total = sum(row["Price"] for row in rows)
        total_matches = total is not None and abs(total - items_total) < 0.5
        receipt_score = ocr_score + 180 * len(rows) + (600 if total_matches else 0)
        candidates.append((receipt_score, lines, vendor, rows, total))

    _, lines, vendor, rows, total = max(candidates, key=lambda candidate: candidate[0])
    return dict(
        vendor=vendor,
        rows=rows,
        total=total,
        date=find_date(lines) if lines else None,
        raw="\n".join(t for t, _ in lines),
        preview=preview,
        quality=quality,
    )

# --- UI SETUP & CUSTOM STYLING ---
st.set_page_config(
    page_title="Snap & Save",
    page_icon=":material/receipt_long:",
    layout="centered",
)

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
    .stMainBlockContainer {
        max-width: 920px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
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
    @media (max-width: 640px) {
        .stMainBlockContainer {
            padding: 0.75rem 0.75rem 2rem;
        }
        .main-header {
            font-size: 2rem;
            margin-bottom: -8px;
        }
        .sub-header {
            margin-bottom: 18px;
            font-size: 0.88rem;
        }
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
tab_cam, tab_up = st.tabs(["Take photo", "Upload batch"])

with tab_cam:
    st.caption(
        "Fill the guide with one clear, well-lit receipt. "
        "The rear camera is preferred automatically."
    )

    camera_result = REAR_CAMERA(
        key=f"rear_camera_{ss.cam_key}",
        on_captured_change=lambda: None,
        width="stretch",
    )
    captured = getattr(camera_result, "captured", None)

    if captured:
        try:
            payload = captured.get("dataUrl", "") if isinstance(captured, dict) else captured
            header, encoded = payload.split(",", 1)
            if not header.startswith("data:image/"):
                raise ValueError("Unexpected camera data")
            b = base64.b64decode(encoded, validate=True)
            if not b or len(b) > 20 * 1024 * 1024:
                raise ValueError("Camera image is empty or too large")
        except (ValueError, TypeError):
            st.error(
                "The camera returned an invalid image. "
                "Please try again or use Upload batch."
            )
        else:
            file_hash = hashlib.md5(b).hexdigest()
            if file_hash not in ss.cam:
                ss.cam[file_hash] = b
                ss.cam_key += 1
                st.rerun()

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
failures = []
progress_bar = st.progress(0.0)
status_text = st.empty()

for i, (h, (name, b)) in enumerate(batch.items(), 1):
    status_text.caption(f"Reading receipt {i} of {len(batch)}...")
    try:
        results[h] = process(b)
    except Exception:
        LOGGER.exception("Failed to process receipt %s", name)
        failures.append(name)
    progress_bar.progress(i / len(batch))

progress_bar.empty()
status_text.empty()

if failures:
    for name in failures:
        st.error(
            f"Could not read {name}. "
            "Retake it in good light or upload a different image."
        )

batch = {h: item for h, item in batch.items() if h in results}
if not batch:
    st.stop()

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
        
        st.image(
            Image.open(io.BytesIO(r["preview"])),
            caption="Automatically detected and straightened scan",
            width="stretch",
        )

        for warning in r["quality"]["warnings"]:
            st.warning(warning, icon=":material/warning:")
        if not r["quality"]["warnings"]:
            st.success(
                "Capture quality looks good and the receipt edges were detected.",
                icon=":material/check_circle:",
            )
        
        if not r["rows"]:
            st.error("No items detected. Try better lighting or add rows below.")
        if r["date"] is None:
            st.info("No date found. Defaults to today.")
            
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            width="stretch",
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
if st.button(
    f"Save {len(all_rows)} item(s)",
    type="primary",
    disabled=all_rows.empty,
    width="stretch",
):
    ss.saved = save(all_rows, ss.fmt)
    st.success(f"Saved successfully to {ss.saved}")

if ss.saved and Path(ss.saved).exists():
    st.download_button(
        label=f"Download {Path(ss.saved).name}",
        data=Path(ss.saved).read_bytes(),
        file_name=Path(ss.saved).name,
        width="stretch",
    )

# Advanced Settings
with st.expander("Advanced Settings"):
    ss.fmt = st.radio("Export Format", ["xlsx", "csv"], index=0, horizontal=True)
    if ss.cam:
        if st.button("Clear Camera Session"):
            ss.cam = {}
            ss.cam_key = 0
            st.rerun()
