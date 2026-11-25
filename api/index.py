#!/usr/bin/env python3
"""
Captcha Solver API — ddddocr + multi-preprocess
================================================
POST /solve
  JSON: { "image": "<base64>" }
  optional: length, min_len, max_len, charset ("alnum"|"alpha"|"digit"|"any")

Response:
  { "ok": true, "text": "ab12xy", "engine": "ddddocr", "ms": 120, "votes": 3 }

Text captcha only. No reCAPTCHA / human services.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

app = Flask(__name__)

# ── ddddocr (lazy init — cold start friendly) ─────────────────
_OCR = None
_OCR_ERR = None


def get_ocr():
    global _OCR, _OCR_ERR
    if _OCR is not None:
        return _OCR
    if _OCR_ERR is not None:
        return None
    try:
        import ddddocr
        _OCR = ddddocr.DdddOcr(show_ad=False)
        return _OCR
    except Exception as e:
        _OCR_ERR = str(e)
        return None


# ── image helpers ─────────────────────────────────────────────

def decode_image(b64: str) -> bytes:
    b64 = (b64 or "").strip()
    if "," in b64 and b64.lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    b64 = re.sub(r"\s+", "", b64)
    return base64.b64decode(b64)


def variants(img_bytes: bytes) -> List[Tuple[str, bytes]]:
    """Multiple preprocess pipelines — vote for higher accuracy."""
    out: List[Tuple[str, bytes]] = []
    base = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def to_bytes(im: Image.Image) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    # 0 raw
    out.append(("raw", img_bytes))

    # 1 grayscale + autocontrast
    g = ImageOps.autocontrast(base.convert("L"))
    out.append(("gray_auto", to_bytes(g.convert("RGB"))))

    # 2 high contrast
    c = ImageEnhance.Contrast(g).enhance(2.4)
    out.append(("contrast", to_bytes(c.convert("RGB"))))

    # 3 sharpen + upscale 2x
    s = c.filter(ImageFilter.SHARPEN)
    s2 = s.resize((s.width * 2, s.height * 2), Image.Resampling.LANCZOS)
    out.append(("sharp_2x", to_bytes(s2.convert("RGB"))))

    # 4 binary threshold 128
    bw = g.point(lambda x: 0 if x < 128 else 255)
    out.append(("bin128", to_bytes(bw.convert("RGB"))))

    # 5 binary threshold 140 + 2x
    bw2 = g.point(lambda x: 0 if x < 140 else 255)
    bw2 = bw2.resize((bw2.width * 2, bw2.height * 2), Image.Resampling.NEAREST)
    out.append(("bin140_2x", to_bytes(bw2.convert("RGB"))))

    # 6 invert (white text on dark)
    inv = ImageOps.invert(g)
    inv = ImageOps.autocontrast(inv)
    out.append(("invert", to_bytes(inv.convert("RGB"))))

    # 7 invert + contrast + 2x
    inv2 = ImageEnhance.Contrast(inv).enhance(2.0)
    inv2 = inv2.resize((inv2.width * 2, inv2.height * 2), Image.Resampling.LANCZOS)
    out.append(("invert_2x", to_bytes(inv2.convert("RGB"))))

    return out


def normalize_text(text: str, charset: str) -> str:
    text = (text or "").strip()
    text = text.replace(" ", "").replace("\n", "")
    if charset == "digit":
        text = re.sub(r"\D", "", text)
    elif charset == "alpha":
        text = re.sub(r"[^A-Za-z]", "", text)
    elif charset == "alnum":
        text = re.sub(r"[^A-Za-z0-9]", "", text)
    # any → light clean only
    else:
        text = re.sub(r"[^A-Za-z0-9]", "", text)
    return text


def length_ok(text: str, min_len: Optional[int], max_len: Optional[int], exact: Optional[int]) -> bool:
    n = len(text)
    if exact is not None:
        return n == exact
    if min_len is not None and n < min_len:
        return False
    if max_len is not None and n > max_len:
        return False
    return n > 0


def solve_bytes(
    img_bytes: bytes,
    charset: str = "alnum",
    min_len: Optional[int] = 3,
    max_len: Optional[int] = 12,
    exact: Optional[int] = None,
) -> Dict[str, Any]:
    ocr = get_ocr()
    if ocr is None:
        return {
            "ok": False,
            "error": f"ddddocr not available: {_OCR_ERR}",
            "engine": None,
        }

    t0 = time.time()
    votes: List[str] = []
    details: List[Dict[str, str]] = []

    for name, raw in variants(img_bytes):
        try:
            pred = ocr.classification(raw)
            text = normalize_text(pred, charset)
            details.append({"variant": name, "raw": pred, "text": text})
            if length_ok(text, min_len, max_len, exact):
                votes.append(text)
        except Exception as e:
            details.append({"variant": name, "error": str(e)})

    ms = int((time.time() - t0) * 1000)

    if not votes:
        # return most common even if length odd
        all_texts = [d["text"] for d in details if d.get("text")]
        if all_texts:
            best = Counter(all_texts).most_common(1)[0][0]
            return {
                "ok": False,
                "error": "no vote matched length constraints",
                "text": best,
                "engine": "ddddocr",
                "ms": ms,
                "votes": 0,
                "debug": details if os.environ.get("DEBUG") == "1" else None,
            }
        return {
            "ok": False,
            "error": "OCR empty",
            "engine": "ddddocr",
            "ms": ms,
            "debug": details if os.environ.get("DEBUG") == "1" else None,
        }

    best, count = Counter(votes).most_common(1)[0]
    return {
        "ok": True,
        "text": best,
        "engine": "ddddocr",
        "ms": ms,
        "votes": count,
        "variants": len(details),
    }


# ── routes ────────────────────────────────────────────────────

@app.get("/")
def index():
    return jsonify(
        {
            "service": "Captcha Solver API",
            "engine": "ddddocr + multi-preprocess vote",
            "endpoint": "POST /solve",
            "body": {
                "image": "base64 string (with or without data:image/...;base64,)",
                "charset": "alnum | alpha | digit | any  (default alnum)",
                "length": "exact length optional",
                "min_len": 3,
                "max_len": 12,
            },
            "example": {
                "image": "iVBORw0KGgoAAAANSUhEUg...",
                "length": 6,
                "charset": "alnum",
            },
        }
    )


@app.get("/health")
def health():
    ocr = get_ocr()
    return jsonify(
        {
            "ok": ocr is not None,
            "engine": "ddddocr",
            "error": _OCR_ERR,
        }
    )


@app.post("/solve")
def solve():
    t_all = time.time()

    # multipart file?
    img_bytes = None
    charset = "alnum"
    exact = None
    min_len = 3
    max_len = 12

    if request.files:
        f = request.files.get("file") or request.files.get("image")
        if f:
            img_bytes = f.read()
        charset = request.form.get("charset", charset)
        if request.form.get("length"):
            exact = int(request.form["length"])
        if request.form.get("min_len"):
            min_len = int(request.form["min_len"])
        if request.form.get("max_len"):
            max_len = int(request.form["max_len"])

    if img_bytes is None:
        data = request.get_json(silent=True) or {}
        b64 = data.get("image") or data.get("base64") or data.get("img")
        if not b64:
            return jsonify({"ok": False, "error": "pass JSON {image: base64} or multipart file"}), 400
        try:
            img_bytes = decode_image(b64)
        except Exception as e:
            return jsonify({"ok": False, "error": f"invalid base64: {e}"}), 400
        charset = data.get("charset", charset)
        if data.get("length") is not None:
            exact = int(data["length"])
        if data.get("min_len") is not None:
            min_len = int(data["min_len"])
        if data.get("max_len") is not None:
            max_len = int(data["max_len"])

    if not img_bytes or len(img_bytes) < 50:
        return jsonify({"ok": False, "error": "image too small / empty"}), 400

    result = solve_bytes(img_bytes, charset=charset, min_len=min_len, max_len=max_len, exact=exact)
    result["total_ms"] = int((time.time() - t_all) * 1000)
    return jsonify(result), (200 if result.get("ok") else 422)


app = app
