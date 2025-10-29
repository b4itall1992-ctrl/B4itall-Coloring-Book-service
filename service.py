import base64, io, os
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import numpy as np, cv2

API_KEY = os.getenv("API_KEY", "")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

LETTER = (2550, 3300)

def _ensure_rgb(img: Image.Image):
    if img.mode in ("RGBA","LA"):
        bg = Image.new("RGBA", img.size, (255,255,255,255))
        bg.paste(img, mask=img.split()[-1])
        return bg.convert("RGB")
    return img.convert("RGB") if img.mode!="RGB" else img

def _auto_upscale(img: Image.Image, min_side=1400):
    w,h = img.size
    if min(w,h) < min_side:
        s = max(1.0, min_side/min(w,h))
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img

def _pad_to_canvas(img: Image.Image, canvas):
    cw,ch = canvas
    w,h = img.size
    s = min(cw/w, ch/h)
    nw,nh = int(w*s), int(h*s)
    img = img.resize((nw,nh), Image.LANCZOS)
    out = Image.new("RGB", (cw,ch), "white")
    out.paste(img, ((cw-nw)//2, (ch-nh)//2))
    return out

def _auto_canny(gray, sigma=0.33):
    v = np.median(gray); lower = max(10, int((1.0 - sigma) * v))
    upper = max(lower+30, int((1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def _remove_small_components(binary, min_area=120):
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(1, nb):
        if stats[i,-1] >= min_area:
            out[labels==i] = 255
    return out

def convert_bytes(image_bytes, line_px=3, dpi=300, despeckle_area=120):
    pil = Image.open(io.BytesIO(image_bytes))
    pil = _ensure_rgb(_auto_upscale(pil, 1400))
    framed = _pad_to_canvas(pil, LETTER)

    g = np.array(ImageOps.grayscale(framed))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); g = clahe.apply(g)
    g_smooth = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)

    ad = cv2.adaptiveThreshold(g_smooth,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,7)
    _, otsu = cv2.threshold(g_smooth,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    silhouettes = cv2.max(255 - ad, 255 - otsu)

    edges = _auto_canny(g_smooth, sigma=0.33)
    merged = cv2.max(edges, silhouettes)

    pruned = cv2.erode(merged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    fg = pruned.copy(); fg[fg>0]=255
    fg = _remove_small_components(fg, min_area=despeckle_area)

    k = max(1, int(round(line_px/2)))
    thick = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),1)

    final_lines = 255 - thick
    L = Image.fromarray(final_lines).convert("L"); L = ImageOps.autocontrast(L)
    out = Image.new("RGB", L.size, "white"); out.paste(L, mask=L)

    png_buf = io.BytesIO(); out.save(png_buf, format="PNG", dpi=(dpi,dpi)); png_buf.seek(0)
    pdf_buf = io.BytesIO(); out.save(pdf_buf, format="PDF", resolution=dpi); pdf_buf.seek(0)
    return png_buf.getvalue(), pdf_buf.getvalue()

@app.get("/health")
def health():
    import cv2, PIL, numpy
    return {"ok": True, "cv2": cv2.__version__}

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    line_px: int = Form(3),
    dpi: int = Form(300),
    despeckle_area: int = Form(120),
    x_api_key: str | None = Header(default=None)
):
    if API_KEY and (x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Bad API key")
    b = await file.read()
    png_bytes, pdf_bytes = convert_bytes(b, line_px, dpi, despeckle_area)
    return JSONResponse({
        "ok": True,
        "png_base64": "data:image/png;base64," + base64.b64encode(png_bytes).decode(),
        "pdf_base64": "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()
    })
