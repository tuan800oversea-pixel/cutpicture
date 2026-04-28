from __future__ import annotations

import base64
import colorsys
import gc
import io
import json
import re
import zipfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from skimage import color


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
STREAMLIT_SAFE_MAX_SIDE = 1600
STREAMLIT_SAFE_TOP_N = 3


@dataclass
class ColorSpec:
    name: str
    hex: str
    rgb: tuple[int, int, int]
    hsl: tuple[int, int, int]
    lab: tuple[float, float, float]
    css: str


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
        }
        img {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", value, flags=re.UNICODE)
    return cleaned.strip("_") or "result"


def read_image_bytes(data: bytes, keep_alpha: bool = True) -> np.ndarray | None:
    if not data:
        return None
    flags = cv2.IMREAD_UNCHANGED if keep_alpha else cv2.IMREAD_COLOR
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)


def read_image_path(path: Path, keep_alpha: bool = True) -> np.ndarray | None:
    if not path.exists():
        return None
    flags = cv2.IMREAD_UNCHANGED if keep_alpha else cv2.IMREAD_COLOR
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def image_to_bytes(img: np.ndarray, ext: str, params: list[int] | None = None) -> bytes:
    ok, buffer = cv2.imencode(ext, img, params or [])
    if not ok:
        raise ValueError(f"无法导出 {ext} 图像")
    return buffer.tobytes()


def ensure_bgr(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def extract_alpha(img: np.ndarray | None) -> np.ndarray | None:
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    alpha = img[:, :, 3]
    if np.max(alpha) == np.min(alpha):
        return None
    return alpha


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def resize_mask_3d(mask_3d: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask_3d[:, :, 0], (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    resized = np.clip(resized, 0.0, 1.0)
    return np.repeat(resized[:, :, np.newaxis], 3, axis=2).astype(np.float32)


def create_low_res_proxy(img: np.ndarray | None, max_width: int = 900) -> np.ndarray | None:
    if img is None:
        return None
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def constrain_image_for_streamlit(img: np.ndarray | None, max_side: int = STREAMLIT_SAFE_MAX_SIDE) -> np.ndarray | None:
    if img is None:
        return None
    kept = np.asarray(img).copy()
    h, w = kept.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return kept
    scale = max_side / float(longest)
    return cv2.resize(
        kept,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def load_uploaded_image(uploaded_file: Any) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    img = read_image_bytes(uploaded_file.getvalue())
    return constrain_image_for_streamlit(img)


def thumbnail_for_ui(img: np.ndarray | None, max_width: int = 240, max_height: int = 320) -> np.ndarray:
    if img is None:
        raise ValueError("无法为 None 图像生成缩略图")
    view = ensure_bgr(img)
    if view is None:
        raise ValueError("图像为空，无法生成缩略图")
    h, w = view.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale >= 1.0:
        return view
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    return cv2.resize(view, (target_w, target_h), interpolation=cv2.INTER_AREA)


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def is_probably_binary(gray: np.ndarray) -> bool:
    sample = gray[:: max(1, gray.shape[0] // 64), :: max(1, gray.shape[1] // 64)]
    return np.unique(sample).size <= 12


def border_background_lab(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    edge = max(8, min(h, w) // 24)
    border = np.concatenate(
        [
            img_bgr[:edge, :, :].reshape(-1, 3),
            img_bgr[-edge:, :, :].reshape(-1, 3),
            img_bgr[:, :edge, :].reshape(-1, 3),
            img_bgr[:, -edge:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    border_lab = cv2.cvtColor(border[np.newaxis, :, :], cv2.COLOR_BGR2LAB)[0]
    return np.median(border_lab, axis=0).astype(np.float32)


def detect_skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 140)) > 0
    skin &= hsv[:, :, 1] > 20
    return skin


def auto_subject_mask(mask_source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src = cv2.resize(mask_source, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    alpha = extract_alpha(src)
    if alpha is not None:
        mask = alpha
    else:
        gray = src if src.ndim == 2 else cv2.cvtColor(ensure_bgr(src), cv2.COLOR_BGR2GRAY)
        if is_probably_binary(gray):
            corner = np.mean(
                np.concatenate(
                    [
                        gray[:32, :32].ravel(),
                        gray[:32, -32:].ravel(),
                        gray[-32:, :32].ravel(),
                        gray[-32:, -32:].ravel(),
                    ]
                )
            )
            center = np.mean(gray[gray.shape[0] // 4 : gray.shape[0] * 3 // 4, gray.shape[1] // 4 : gray.shape[1] * 3 // 4])
            thresh_type = cv2.THRESH_BINARY_INV if corner > center else cv2.THRESH_BINARY
            _, mask = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        else:
            bgr = ensure_bgr(src)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            bg_lab = border_background_lab(bgr)
            dist = np.linalg.norm(lab - bg_lab, axis=2)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            dist_thresh = max(10.0, float(np.percentile(dist, 82)) * 0.55)
            mask_bool = (dist > dist_thresh) | (gray < 245) | (hsv[:, :, 1] > 28)
            mask = (mask_bool.astype(np.uint8) * 255)
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    if np.count_nonzero(mask) > 0:
        mask = largest_component(mask)
    return mask


def preprocess_mask(mask_source: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask_source is None:
        return np.zeros((*shape, 3), dtype=np.float32)
    mask = auto_subject_mask(mask_source, shape)
    if np.count_nonzero(mask) == 0:
        return np.zeros((*shape, 3), dtype=np.float32)
    mask = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0
    return np.repeat(mask[:, :, np.newaxis], 3, axis=2)


def sample_pixels(pixels: np.ndarray, max_samples: int = 12000) -> np.ndarray:
    if pixels.shape[0] <= max_samples:
        return pixels
    idx = np.linspace(0, pixels.shape[0] - 1, max_samples, dtype=np.int32)
    return pixels[idx]


def dominant_lab_8bit_from_pixels(lab_pixels: np.ndarray) -> np.ndarray:
    if lab_pixels.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    pixels = np.float32(sample_pixels(lab_pixels.reshape(-1, 3)))
    if pixels.shape[0] < 32:
        return np.mean(pixels, axis=0)
    clusters = min(4, max(2, pixels.shape[0] // 2500 + 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 18, 0.25)
    _, labels, centers = cv2.kmeans(pixels, clusters, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=clusters)
    chroma = np.linalg.norm(centers[:, 1:] - 128.0, axis=1)
    light_penalty = np.abs(centers[:, 0] - 132.0) * 0.08
    score = counts + (chroma * 0.35) - light_penalty
    return centers[int(np.argmax(score))]


def lab8_to_std(lab_8bit: np.ndarray) -> np.ndarray:
    lab_8bit = np.asarray(lab_8bit, dtype=np.float32)
    return np.array(
        [
            lab_8bit[0] * (100.0 / 255.0),
            lab_8bit[1] - 128.0,
            lab_8bit[2] - 128.0,
        ],
        dtype=np.float32,
    )


def extract_masked_mean_std_lab(img_bgr: np.ndarray, mask_3d: np.ndarray | None = None) -> np.ndarray:
    img_f = ensure_bgr(img_bgr).astype(np.float32) / 255.0
    lab_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)
    if mask_3d is not None:
        mask_bool = mask_3d[:, :, 0] > 0.08
        if np.any(mask_bool):
            return np.mean(lab_f[mask_bool], axis=0).astype(np.float32)
    h, w = img_f.shape[:2]
    crop = lab_f[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    return np.mean(crop.reshape(-1, 3), axis=0).astype(np.float32)


def std_lab_to_rgb(std_lab: np.ndarray) -> tuple[int, int, int]:
    lab = np.asarray(std_lab, dtype=np.float32).reshape(1, 1, 3)
    rgb = color.lab2rgb(lab)[0, 0]
    rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def build_color_spec(name: str, std_lab: np.ndarray) -> ColorSpec:
    rgb = std_lab_to_rgb(std_lab)
    hex_value = "#{:02X}{:02X}{:02X}".format(*rgb)
    h, l, s = colorsys.rgb_to_hls(*(channel / 255.0 for channel in rgb))
    hsl = (int(round(h * 360)), int(round(s * 100)), int(round(l * 100)))
    text_color = "#111111" if l > 0.62 else "#F8F8F8"
    css = f"background:{hex_value};color:{text_color};padding:12px 16px;border-radius:12px;font-weight:600;"
    return ColorSpec(
        name=name,
        hex=hex_value,
        rgb=rgb,
        hsl=hsl,
        lab=(round(float(std_lab[0]), 2), round(float(std_lab[1]), 2), round(float(std_lab[2]), 2)),
        css=css,
    )


def create_color_chip(rgb: tuple[int, int, int], size: tuple[int, int] = (220, 120), text: str | None = None) -> np.ndarray:
    chip = np.full((size[1], size[0], 3), rgb[::-1], dtype=np.uint8)
    border = (32, 32, 32) if sum(rgb) > 380 else (245, 245, 245)
    cv2.rectangle(chip, (0, 0), (size[0] - 1, size[1] - 1), border, 2)
    if text:
        cv2.putText(chip, text, (12, size[1] // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, border, 2, cv2.LINE_AA)
    return chip


def create_focus_preview(img_bgr: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    mask = np.clip(mask_3d[:, :, 0], 0.0, 1.0)[:, :, np.newaxis]
    background = np.full_like(img_bgr, 248)
    focused = img_bgr.astype(np.float32) * mask + background.astype(np.float32) * (1.0 - mask)
    outline = cv2.Canny((mask[:, :, 0] * 255).astype(np.uint8), 40, 120)
    focused = np.clip(focused, 0, 255).astype(np.uint8)
    focused[outline > 0] = (32, 96, 220)
    return focused


def masked_blend(base_img: np.ndarray, smooth_img: np.ndarray, mask_3d: np.ndarray, strength: float) -> np.ndarray:
    alpha = np.clip(mask_3d[:, :, 0] * strength, 0.0, 1.0)
    if base_img.ndim == 3:
        alpha = alpha[:, :, np.newaxis]
    blended = base_img.astype(np.float32) * (1.0 - alpha) + smooth_img.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def blend_with_alpha(base_img: np.ndarray, smooth_img: np.ndarray, alpha_map: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha_map, 0.0, 1.0)
    if base_img.ndim == 3 and alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]
    blended = base_img.astype(np.float32) * (1.0 - alpha) + smooth_img.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def pale_color_strength(target_lab: np.ndarray) -> float:
    l_t, a_t, b_t = target_lab.astype(float)
    chroma = float(np.linalg.norm([a_t - 128.0, b_t - 128.0]))
    light_factor = np.clip((l_t - 168.0) / 58.0, 0.0, 1.0)
    chroma_factor = 1.0 - np.clip(chroma / 42.0, 0.0, 1.0)
    return float(light_factor * chroma_factor)


def dark_color_strength(target_lab: np.ndarray) -> float:
    l_t = float(target_lab[0])
    return float(np.clip((118.0 - l_t) / 64.0, 0.0, 1.0))


def white_color_strength(target_lab: np.ndarray) -> float:
    l_t, a_t, b_t = target_lab.astype(float)
    chroma = float(np.linalg.norm([a_t - 128.0, b_t - 128.0]))
    light_factor = np.clip((l_t - 205.0) / 40.0, 0.0, 1.0)
    chroma_factor = 1.0 - np.clip(chroma / 20.0, 0.0, 1.0)
    return float(light_factor * chroma_factor)


def bright_flat_strength(target_lab: np.ndarray) -> float:
    l_t = float(target_lab[0])
    return float(np.clip((l_t - 178.0) / 44.0, 0.0, 1.0))


def neon_color_strength(target_lab: np.ndarray) -> float:
    _, a_t, b_t = target_lab.astype(float)
    chroma = float(np.linalg.norm([a_t - 128.0, b_t - 128.0]))
    return float(np.clip((chroma - 28.0) / 38.0, 0.0, 1.0))


def classify_target_style(target_lab: np.ndarray) -> str:
    white_strength = white_color_strength(target_lab)
    bright_strength = bright_flat_strength(target_lab)
    pale_strength = pale_color_strength(target_lab)
    dark_strength = dark_color_strength(target_lab)
    neon_strength = neon_color_strength(target_lab)
    if neon_strength >= 0.62:
        return "neon"
    if white_strength >= 0.42:
        return "white"
    if pale_strength >= 0.35:
        return "light"
    if dark_strength >= 0.40:
        return "dark"
    return "normal"


def build_reference_mask(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    y, x = np.indices((h, w))
    center_ellipse = (((x - w / 2) / (w * 0.36)) ** 2 + ((y - h / 2) / (h * 0.40)) ** 2) <= 1.0
    skin = detect_skin_mask(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg_lab = border_background_lab(img_bgr)
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    dynamic = max(8.0, float(np.percentile(dist[center_ellipse], 60)) * 0.75)
    garment = center_ellipse & (~skin) & ((dist > dynamic) | (gray < 248) | (hsv[:, :, 1] > 18))
    if np.count_nonzero(garment) < h * w * 0.02:
        garment = center_ellipse & (~skin)
    if np.count_nonzero(garment) < h * w * 0.01:
        garment = center_ellipse
    mask = (garment.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if np.count_nonzero(mask) > 0:
        mask = largest_component(mask)
    return np.repeat((mask.astype(np.float32) / 255.0)[:, :, np.newaxis], 3, axis=2)


def extract_region_lab_8bit(img_bgr: np.ndarray, mask_3d: np.ndarray | None = None) -> np.ndarray:
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if mask_3d is not None:
        mask_bool = mask_3d[:, :, 0] > 0.08
        if np.any(mask_bool):
            return dominant_lab_8bit_from_pixels(img_lab[mask_bool])
    h, w = img_bgr.shape[:2]
    crop = img_lab[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    return dominant_lab_8bit_from_pixels(crop.reshape(-1, 3))


def extract_region_std_lab(img_bgr: np.ndarray, mask_3d: np.ndarray | None = None) -> np.ndarray:
    return extract_masked_mean_std_lab(img_bgr, mask_3d)


def analyze_validation_reference_image(ref_img: np.ndarray, label: str) -> dict[str, Any]:
    bgr = ensure_bgr(ref_img)
    mask_3d = build_reference_mask(bgr)
    std_lab = extract_masked_mean_std_lab(bgr, mask_3d)
    fallback_render_lab = extract_region_lab_8bit(bgr, mask_3d)
    spec = build_color_spec(label, std_lab)
    return {
        "label": label,
        "image_bgr": bgr,
        "mask_3d": mask_3d,
        "std_lab": std_lab.astype(np.float32),
        "fallback_render_lab": fallback_render_lab.astype(np.float32),
        "spec": spec,
        "chip_bgr": create_color_chip(spec.rgb, text=spec.hex),
        "focus_bgr": create_focus_preview(bgr, mask_3d),
    }


def analyze_render_reference_image(ref_img: np.ndarray, label: str) -> dict[str, Any]:
    bgr = ensure_bgr(ref_img)
    render_lab = extract_region_lab_8bit(bgr)
    render_std_lab = lab8_to_std(render_lab)
    render_spec = build_color_spec(f"{label}_render", render_std_lab)
    return {
        "image_bgr": bgr,
        "render_lab": render_lab.astype(np.float32),
        "render_spec": render_spec,
        "chip_bgr": create_color_chip(render_spec.rgb, text=render_spec.hex),
    }


def analyze_target_input(target_input: dict[str, Any]) -> dict[str, Any]:
    label = target_input["label"]
    validation = analyze_validation_reference_image(target_input["validation_image"], label)
    render_image = target_input.get("render_image")
    if render_image is not None:
        render_source = analyze_render_reference_image(render_image, label)
        render_lab = render_source["render_lab"]
        render_source_kind = "纯色色块图"
        render_source_bgr = render_source["image_bgr"]
        render_source_chip_bgr = render_source["chip_bgr"]
        render_source_hex = render_source["render_spec"].hex
    else:
        render_lab = validation["fallback_render_lab"]
        render_source_kind = "校验模特图回退"
        render_source_bgr = validation["image_bgr"]
        render_source_chip_bgr = validation["chip_bgr"]
        render_source_hex = validation["spec"].hex
    style = classify_target_style(render_lab)
    return {
        "label": label,
        "image_bgr": validation["image_bgr"],
        "mask_3d": validation["mask_3d"],
        "render_lab": render_lab.astype(np.float32),
        "std_lab": validation["std_lab"],
        "spec": validation["spec"],
        "style": style,
        "chip_bgr": validation["chip_bgr"],
        "focus_bgr": validation["focus_bgr"],
        "validation_source_kind": "校验模特图",
        "render_source_kind": render_source_kind,
        "render_source_bgr": render_source_bgr,
        "render_source_chip_bgr": render_source_chip_bgr,
        "render_source_hex": render_source_hex,
        "has_render_swatch": render_image is not None,
    }


def render_standard(orig_img: np.ndarray, gray_img: np.ndarray, mask_3d: np.ndarray, target_lab: np.ndarray, params: tuple[float, float, float, float, float]) -> np.ndarray:
    gamma, l_off, a_off, b_off, detail_boost = params
    l_t, a_t, b_t = target_lab.astype(float)
    pale_strength = pale_color_strength(target_lab)
    dark_strength = dark_color_strength(target_lab)
    white_strength = white_color_strength(target_lab)
    bright_strength = bright_flat_strength(target_lab)
    bright_strength = bright_flat_strength(target_lab)
    denoise_d = 3 if dark_strength > 0.35 else (7 if pale_strength >= 0.2 else 5)
    sigma_color = 16 + int(round(30 * pale_strength - 4 * dark_strength + 18 * white_strength))
    sigma_space = 16 + int(round(22 * pale_strength - 4 * dark_strength + 16 * white_strength))
    sigma_color = max(10, sigma_color)
    sigma_space = max(10, sigma_space)
    denoised_gray = cv2.bilateralFilter(gray_img, d=denoise_d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    blur_layer = cv2.GaussianBlur(denoised_gray, (15, 15), 0)
    detail_gain = detail_boost * (1.0 - 0.58 * pale_strength - 0.22 * white_strength - 0.06 * dark_strength)
    detail_layer = cv2.subtract(denoised_gray, blur_layer).astype(np.float32) * detail_gain
    img_norm = denoised_gray.astype(np.float32) / 255.0
    img_gamma = np.power(img_norm + 1e-7, 1.0 / max(gamma, 0.01))
    mask_bool = mask_3d[:, :, 0] > 0.08
    structure_mask = np.zeros_like(gray_img, dtype=np.float32)
    if np.any(mask_bool):
        detail_abs = np.abs(detail_layer[mask_bool])
        detail_scale = float(np.percentile(detail_abs, 88)) + 1e-6
        grad_x = cv2.Sobel(denoised_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(denoised_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_scale = float(np.percentile(grad_mag[mask_bool], 88)) + 1e-6
        structure_mask = np.clip((np.abs(detail_layer) / detail_scale) * 0.58 + (grad_mag / grad_scale) * 0.42, 0.0, 1.0)
        structure_mask = cv2.GaussianBlur(structure_mask, (7, 7), 0)
        if dark_strength > 0.02:
            flat_guard = 1.0 - np.clip(structure_mask, 0.0, 1.0)
            detail_layer *= 1.0 - (0.48 * dark_strength * flat_guard)
    flat_mask = np.clip(mask_3d[:, :, 0] * (1.0 - np.clip(structure_mask, 0.0, 1.0)), 0.0, 1.0)
    current_mean_l = float(np.mean(img_gamma[mask_bool])) if np.any(mask_bool) else 0.5
    target_l = np.clip(l_t + l_off, 0, 255)
    shift_l = (target_l / 255.0) - current_mean_l
    shadow_map = np.clip((img_gamma + shift_l) * 255.0 + detail_layer, 0, 255.0)
    if white_strength > 0.03:
        highlight_map = np.clip((shadow_map - 118.0) / 110.0, 0.0, 1.0) * (1.0 - 0.55 * structure_mask)
        shadow_map = np.clip(shadow_map + highlight_map * (7.0 + 12.0 * white_strength), 0.0, 255.0)
    if pale_strength > 0.01:
        shadow_u8 = shadow_map.astype(np.uint8)
        smooth_shadow = cv2.bilateralFilter(
            shadow_u8,
            d=7,
            sigmaColor=10 + int(round(18 * pale_strength + 22 * white_strength)),
            sigmaSpace=14 + int(round(18 * pale_strength + 18 * white_strength)),
        )
        shadow_alpha = np.clip(mask_3d[:, :, 0] * (0.18 + 0.34 * pale_strength + 0.18 * white_strength), 0.0, 1.0)
        shadow_alpha *= (1.0 - 0.78 * structure_mask)
        shadow_map = blend_with_alpha(
            shadow_u8,
            smooth_shadow,
            shadow_alpha,
        ).astype(np.float32)
    final_a = np.clip(a_t + a_off, 0, 255)
    final_b = np.clip(b_t + b_off, 0, 255)
    if white_strength > 0.02:
        final_a = 128.0 + (final_a - 128.0) * (1.0 - 0.38 * white_strength)
        final_b = 128.0 + (final_b - 128.0) * (1.0 - 0.38 * white_strength)
    merged_lab = cv2.merge(
        [
            shadow_map.astype(np.uint8),
            np.full_like(shadow_map, int(final_a), dtype=np.uint8),
            np.full_like(shadow_map, int(final_b), dtype=np.uint8),
        ]
    )
    result_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    if dark_strength > 0.05:
        dark_bilateral = cv2.bilateralFilter(
            result_bgr,
            d=7,
            sigmaColor=12 + int(round(24 * dark_strength)),
            sigmaSpace=14 + int(round(22 * dark_strength)),
        )
        dark_median = cv2.medianBlur(result_bgr, 3)
        dark_smooth = cv2.addWeighted(dark_bilateral, 0.72, dark_median, 0.28, 0)
        dark_alpha = np.clip(flat_mask * (0.10 + 0.34 * dark_strength), 0.0, 1.0)
        result_bgr = blend_with_alpha(result_bgr, dark_smooth, dark_alpha)
    if pale_strength > 0.01:
        smooth_result = cv2.bilateralFilter(
            result_bgr,
            d=7,
            sigmaColor=14 + int(round(24 * pale_strength + 24 * white_strength)),
            sigmaSpace=16 + int(round(20 * pale_strength + 22 * white_strength)),
        )
        result_alpha = np.clip(mask_3d[:, :, 0] * (0.12 + 0.30 * pale_strength + 0.16 * white_strength), 0.0, 1.0)
        result_alpha *= (1.0 - 0.80 * structure_mask)
        result_bgr = blend_with_alpha(
            result_bgr,
            smooth_result,
            result_alpha,
        )
    if white_strength > 0.04:
        result_lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask_alpha = np.clip(mask_3d[:, :, 0], 0.0, 1.0)
        preserve = 1.0 - 0.86 * structure_mask
        ab_alpha = mask_alpha * (0.18 + 0.28 * white_strength) * preserve
        l_alpha = mask_alpha * (0.08 + 0.14 * white_strength) * preserve
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0] + l_alpha * (6.0 + 10.0 * white_strength), 0.0, 255.0)
        result_lab[:, :, 1] = result_lab[:, :, 1] * (1.0 - ab_alpha) + final_a * ab_alpha
        result_lab[:, :, 2] = result_lab[:, :, 2] * (1.0 - ab_alpha) + final_b * ab_alpha
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    final_f = (result_bgr.astype(np.float32) / 255.0) * mask_3d + (orig_img.astype(np.float32) / 255.0) * (1.0 - mask_3d)
    return (np.clip(final_f, 0, 1) * 255.0).astype(np.uint8)


def cleanup_dark_flat_noise(result_bgr: np.ndarray, flat_mask: np.ndarray, structure_mask: np.ndarray, dark_strength: float) -> np.ndarray:
    if dark_strength <= 0.04:
        return result_bgr
    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0].astype(np.uint8)
    a_channel = lab[:, :, 1].astype(np.uint8)
    b_channel = lab[:, :, 2].astype(np.uint8)
    chroma_soft = cv2.GaussianBlur(cv2.merge([a_channel, b_channel]), (0, 0), 1.25 + 0.85 * dark_strength)
    l_low = cv2.resize(l_channel, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    l_low = cv2.GaussianBlur(l_low, (0, 0), 1.1 + 1.2 * dark_strength)
    l_smooth = cv2.resize(l_low, (l_channel.shape[1], l_channel.shape[0]), interpolation=cv2.INTER_CUBIC)
    local_noise = cv2.absdiff(l_channel, cv2.GaussianBlur(l_channel, (0, 0), 1.0))
    noise_boost = 0.0
    noise_mask = flat_mask > 0.08
    if np.any(noise_mask):
        noise_level = float(np.percentile(local_noise[noise_mask], 82))
        noise_boost = float(np.clip((noise_level - 2.4) / 7.5, 0.0, 1.0))
    preserve = 1.0 - 0.35 * np.clip(structure_mask, 0.0, 1.0)
    chroma_alpha = np.clip(flat_mask * (0.18 + 0.36 * dark_strength + 0.22 * noise_boost) * preserve, 0.0, 1.0)
    l_alpha = np.clip(flat_mask * (0.10 + 0.24 * dark_strength + 0.20 * noise_boost) * preserve, 0.0, 1.0)
    lab[:, :, 0] = lab[:, :, 0] * (1.0 - l_alpha) + l_smooth.astype(np.float32) * l_alpha
    lab[:, :, 1] = lab[:, :, 1] * (1.0 - chroma_alpha) + chroma_soft[:, :, 0].astype(np.float32) * chroma_alpha
    lab[:, :, 2] = lab[:, :, 2] * (1.0 - chroma_alpha) + chroma_soft[:, :, 1].astype(np.float32) * chroma_alpha
    return cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)


def cleanup_light_flat_noise(
    result_bgr: np.ndarray,
    flat_mask: np.ndarray,
    structure_mask: np.ndarray,
    pale_strength: float,
    white_strength: float,
) -> np.ndarray:
    strength = max(float(pale_strength), float(white_strength) * 0.9)
    if strength <= 0.03:
        return result_bgr
    nlm = cv2.fastNlMeansDenoisingColored(
        result_bgr,
        None,
        h=max(4, int(round(5 + 8 * strength))),
        hColor=max(5, int(round(7 + 10 * strength))),
        templateWindowSize=7,
        searchWindowSize=21,
    )
    lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0].astype(np.uint8)
    ab = lab[:, :, 1:].astype(np.uint8)
    low_l = cv2.resize(l_channel, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    low_l = cv2.GaussianBlur(low_l, (0, 0), 1.2 + 1.0 * strength)
    smooth_l = cv2.resize(low_l, (l_channel.shape[1], l_channel.shape[0]), interpolation=cv2.INTER_CUBIC)
    smooth_ab = cv2.GaussianBlur(ab, (0, 0), 1.0 + 1.15 * strength)
    local_noise = cv2.absdiff(l_channel, cv2.GaussianBlur(l_channel, (0, 0), 0.9))
    noise_boost = 0.0
    flat_bool = flat_mask > 0.08
    if np.any(flat_bool):
        noise_level = float(np.percentile(local_noise[flat_bool], 84))
        noise_boost = float(np.clip((noise_level - 1.8) / 6.0, 0.0, 1.0))
    preserve = 1.0 - 0.68 * np.clip(structure_mask, 0.0, 1.0)
    l_alpha = np.clip(flat_mask * (0.16 + 0.34 * strength + 0.22 * noise_boost) * preserve, 0.0, 1.0)
    ab_alpha = np.clip(flat_mask * (0.24 + 0.40 * strength + 0.24 * noise_boost) * preserve, 0.0, 1.0)
    lab[:, :, 0] = lab[:, :, 0] * (1.0 - l_alpha) + smooth_l.astype(np.float32) * l_alpha
    lab[:, :, 1] = lab[:, :, 1] * (1.0 - ab_alpha) + smooth_ab[:, :, 0].astype(np.float32) * ab_alpha
    lab[:, :, 2] = lab[:, :, 2] * (1.0 - ab_alpha) + smooth_ab[:, :, 1].astype(np.float32) * ab_alpha
    cleaned = cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)
    flat_smooth = cv2.bilateralFilter(
        cleaned,
        d=9,
        sigmaColor=22 + int(round(18 * strength + 16 * noise_boost)),
        sigmaSpace=18 + int(round(16 * strength + 14 * noise_boost)),
    )
    flat_smooth = cv2.GaussianBlur(flat_smooth, (0, 0), 0.9 + 0.9 * strength)
    macro_low = cv2.resize(cleaned, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
    macro_low = cv2.GaussianBlur(macro_low, (0, 0), 1.1 + 1.4 * strength)
    macro_smooth = cv2.resize(macro_low, (cleaned.shape[1], cleaned.shape[0]), interpolation=cv2.INTER_CUBIC)
    nlm_alpha = np.clip(flat_mask * (0.08 + 0.26 * strength + 0.22 * noise_boost) * preserve, 0.0, 1.0)
    cleaned = blend_with_alpha(cleaned, nlm, nlm_alpha)
    flat_alpha = np.clip(flat_mask * (0.18 + 0.52 * strength + 0.30 * noise_boost) * preserve, 0.0, 1.0)
    cleaned = blend_with_alpha(cleaned, flat_smooth, flat_alpha)
    macro_alpha = np.clip(flat_mask * (0.10 + 0.42 * strength + 0.24 * noise_boost) * (1.0 - 0.78 * np.clip(structure_mask, 0.0, 1.0)), 0.0, 1.0)
    return blend_with_alpha(cleaned, macro_smooth, macro_alpha)


def cleanup_vivid_flat_noise(
    result_bgr: np.ndarray,
    flat_mask: np.ndarray,
    structure_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    if strength <= 0.03:
        return result_bgr
    smooth = cv2.bilateralFilter(
        result_bgr,
        d=9,
        sigmaColor=20 + int(round(16 * strength)),
        sigmaSpace=18 + int(round(14 * strength)),
    )
    low = cv2.resize(smooth, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    low = cv2.GaussianBlur(low, (0, 0), 1.0 + 1.2 * strength)
    macro = cv2.resize(low, (result_bgr.shape[1], result_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    preserve = 1.0 - 0.76 * np.clip(structure_mask, 0.0, 1.0)
    alpha = np.clip(flat_mask * (0.18 + 0.46 * strength) * preserve, 0.0, 1.0)
    return blend_with_alpha(result_bgr, macro, alpha)


def render_neon(orig_img: np.ndarray, mask_3d: np.ndarray, target_lab: np.ndarray, params: tuple[float, float, float, float]) -> np.ndarray:
    l_off, a_off, b_off, detail_gain = params
    l_t, a_t, b_t = target_lab.astype(float)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.bilateralFilter(gray, d=7, sigmaColor=18, sigmaSpace=18)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_detail = clahe.apply(denoised_gray).astype(np.float32) / 255.0
    mask_bool = mask_3d[:, :, 0] > 0.08
    avg_detail = float(np.mean(gray_detail[mask_bool])) if np.any(mask_bool) else 0.5
    detail_map = gray_detail - avg_detail
    detail_abs = np.abs(detail_map)
    detail_scale = float(np.percentile(detail_abs[mask_bool], 88)) + 1e-6 if np.any(mask_bool) else 1.0
    grad_x = cv2.Sobel(denoised_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(denoised_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_scale = float(np.percentile(grad_mag[mask_bool], 88)) + 1e-6 if np.any(mask_bool) else 1.0
    structure_mask = np.clip((detail_abs / detail_scale) * 0.56 + (grad_mag / grad_scale) * 0.44, 0.0, 1.0)
    structure_mask = cv2.GaussianBlur(structure_mask, (7, 7), 0)
    bright_strength = bright_flat_strength(target_lab)
    flat_mask = np.clip(mask_3d[:, :, 0] * (1.0 - np.clip(structure_mask, 0.0, 1.0)), 0.0, 1.0)
    detail_gain_effective = detail_gain * (0.72 - 0.34 * bright_strength)
    target_l = np.clip(l_t + l_off, 0, 255)
    target_a = np.clip(a_t + a_off, 0, 255)
    target_b = np.clip(b_t + b_off, 0, 255)
    l_layer = np.clip(np.full(gray.shape, target_l, dtype=np.float32) + (detail_map * detail_gain_effective), 0, 255)
    final_lab = cv2.merge(
        [
            l_layer.astype(np.uint8),
            np.full(gray.shape, int(target_a), dtype=np.uint8),
            np.full(gray.shape, int(target_b), dtype=np.uint8),
        ]
    )
    res_bgr = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
    smooth = cv2.bilateralFilter(res_bgr, d=9, sigmaColor=28 + int(round(12 * bright_strength)), sigmaSpace=24 + int(round(10 * bright_strength)))
    smooth = cv2.GaussianBlur(smooth, (0, 0), 0.9 + 0.8 * bright_strength)
    smooth_alpha = np.clip(flat_mask * (0.22 + 0.42 * bright_strength), 0.0, 1.0)
    res_bgr = blend_with_alpha(res_bgr, smooth, smooth_alpha)
    if bright_strength > 0.05:
        res_bgr = cleanup_vivid_flat_noise(res_bgr, flat_mask, structure_mask, bright_strength)
    final_out = res_bgr.astype(np.float32) * mask_3d + orig_img.astype(np.float32) * (1.0 - mask_3d)
    return np.clip(final_out, 0, 255).astype(np.uint8)


def render_region(orig_img: np.ndarray, gray_img: np.ndarray, mask_3d: np.ndarray, target_lab: np.ndarray, params: tuple[float, ...], is_neon: bool) -> np.ndarray:
    if is_neon:
        return render_neon(orig_img, mask_3d, target_lab, params)  # type: ignore[arg-type]
    return render_standard(orig_img, gray_img, mask_3d, target_lab, params)  # type: ignore[arg-type]


def evaluate_delta_e(result_bgr: np.ndarray, mask_3d: np.ndarray, target_std_lab: np.ndarray) -> tuple[float, np.ndarray]:
    current_lab_std = extract_region_std_lab(result_bgr, mask_3d)
    de = float(color.deltaE_ciede2000(target_std_lab, current_lab_std))
    return de, current_lab_std


def clamp_standard_params(params: tuple[float, ...]) -> tuple[float, float, float, float, float]:
    gamma, l_off, a_off, b_off, detail = params
    return (
        float(np.clip(gamma, 0.88, 1.15)),
        float(np.clip(l_off, -84, 84)),
        float(np.clip(a_off, -42, 42)),
        float(np.clip(b_off, -42, 42)),
        float(np.clip(detail, 0.95, 1.7)),
    )


def clamp_neon_params(params: tuple[float, ...]) -> tuple[float, float, float, float]:
    l_off, a_off, b_off, detail = params
    return (
        float(np.clip(l_off, -84, 84)),
        float(np.clip(a_off, -48, 48)),
        float(np.clip(b_off, -48, 48)),
        float(np.clip(detail, 90, 155)),
    )


def unique_best_candidates(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cand in sorted(candidates, key=lambda item: item["de"]):
        key = tuple(round(float(p), 3 if idx == 0 else 2) for idx, p in enumerate(cand["params"]))
        if key in seen:
            continue
        if unique and abs(cand["de"] - unique[-1]["de"]) < 0.03:
            continue
        seen.add(key)
        unique.append(cand)
        if len(unique) >= limit:
            break
    return unique


def optimize_region_candidates(orig_hr: np.ndarray, mask_3d_hr: np.ndarray, target: dict[str, Any], label: str, top_n: int = 4) -> list[dict[str, Any]]:
    orig_lr = create_low_res_proxy(orig_hr, 900)
    if orig_lr is None:
        return []
    mask_3d_lr = resize_mask_3d(mask_3d_hr, orig_lr.shape[:2])
    gray_lr = cv2.cvtColor(orig_lr, cv2.COLOR_BGR2GRAY)
    gray_hr = cv2.cvtColor(orig_hr, cv2.COLOR_BGR2GRAY)
    render_lab = target["render_lab"]
    std_lab = target["std_lab"]
    is_neon = abs(float(render_lab[1]) - 128.0) > 26 or abs(float(render_lab[2]) - 128.0) > 28
    low_candidates: list[dict[str, Any]] = []

    if is_neon:
        seeds: list[tuple[float, ...]] = [
            (0.0, 0.0, 0.0, 118.0),
            (-6.0, 0.0, 0.0, 130.0),
            (6.0, 0.0, 0.0, 102.0),
        ]
    else:
        seeds = [
            (1.00, 0.0, 0.0, 0.0, 1.22),
            (0.96, 0.0, 0.0, 0.0, 1.10),
            (1.05, 0.0, 0.0, 0.0, 1.34),
        ]

    for seed in seeds:
        params = seed
        learning_rate = 0.56
        for _ in range(11):
            preview = render_region(orig_lr, gray_lr, mask_3d_lr, render_lab, params, is_neon)
            de, current_std_lab = evaluate_delta_e(preview, mask_3d_lr, std_lab)
            low_candidates.append({"params": params, "de": de, "lab": current_std_lab, "label": label})
            err = std_lab - current_std_lab
            if is_neon:
                params = clamp_neon_params(
                    (
                        params[0] + float(err[0] * 2.2 * learning_rate),
                        params[1] + float(err[1] * learning_rate),
                        params[2] + float(err[2] * learning_rate),
                        params[3] + float((abs(err[1]) + abs(err[2])) * 0.6 - de * 0.12),
                    )
                )
            else:
                params = clamp_standard_params(
                    (
                        params[0] + float(np.clip(-err[0] * 0.002, -0.025, 0.025)),
                        params[1] + float(err[0] * 2.35 * learning_rate),
                        params[2] + float(err[1] * learning_rate),
                        params[3] + float(err[2] * learning_rate),
                        params[4] + (0.06 if de > 7 else -0.03),
                    )
                )
            learning_rate *= 0.9

    seed_best = unique_best_candidates(low_candidates, 3)
    for best in seed_best:
        base = best["params"]
        if is_neon:
            for dl, da, db in product((-6.0, 0.0, 6.0), (-5.0, 0.0, 5.0), (-5.0, 0.0, 5.0)):
                trial = clamp_neon_params((base[0] + dl, base[1] + da, base[2] + db, base[3]))
                preview = render_region(orig_lr, gray_lr, mask_3d_lr, render_lab, trial, is_neon)
                de, current_std_lab = evaluate_delta_e(preview, mask_3d_lr, std_lab)
                low_candidates.append({"params": trial, "de": de, "lab": current_std_lab, "label": label})
            best_local = min(low_candidates, key=lambda item: item["de"])
            base = best_local["params"]
            for dd in (-18.0, 0.0, 18.0):
                trial = clamp_neon_params((base[0], base[1], base[2], base[3] + dd))
                preview = render_region(orig_lr, gray_lr, mask_3d_lr, render_lab, trial, is_neon)
                de, current_std_lab = evaluate_delta_e(preview, mask_3d_lr, std_lab)
                low_candidates.append({"params": trial, "de": de, "lab": current_std_lab, "label": label})
        else:
            for dl, da, db in product((-7.0, 0.0, 7.0), (-4.0, 0.0, 4.0), (-4.0, 0.0, 4.0)):
                trial = clamp_standard_params((base[0], base[1] + dl, base[2] + da, base[3] + db, base[4]))
                preview = render_region(orig_lr, gray_lr, mask_3d_lr, render_lab, trial, is_neon)
                de, current_std_lab = evaluate_delta_e(preview, mask_3d_lr, std_lab)
                low_candidates.append({"params": trial, "de": de, "lab": current_std_lab, "label": label})
            local_best = min(low_candidates, key=lambda item: item["de"])
            base = local_best["params"]
            for dg, dd in product((-0.03, 0.0, 0.03), (-0.10, 0.0, 0.10)):
                trial = clamp_standard_params((base[0] + dg, base[1], base[2], base[3], base[4] + dd))
                preview = render_region(orig_lr, gray_lr, mask_3d_lr, render_lab, trial, is_neon)
                de, current_std_lab = evaluate_delta_e(preview, mask_3d_lr, std_lab)
                low_candidates.append({"params": trial, "de": de, "lab": current_std_lab, "label": label})

    final_low = unique_best_candidates(low_candidates, max(top_n + 2, 5))
    final_hr: list[dict[str, Any]] = []
    for cand in final_low:
        img_hr = render_region(orig_hr, gray_hr, mask_3d_hr, render_lab, cand["params"], is_neon)
        de, current_std_lab = evaluate_delta_e(img_hr, mask_3d_hr, std_lab)
        final_hr.append(
            {
                "params": cand["params"],
                "de": de,
                "lab": current_std_lab,
                "image": img_hr,
                "label": label,
                "is_neon": is_neon,
            }
        )
    return unique_best_candidates(final_hr, top_n)


def composite_with_mask(base_img: np.ndarray, layer_img: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    base_f = base_img.astype(np.float32)
    layer_f = layer_img.astype(np.float32)
    merged = layer_f * mask_3d + base_f * (1.0 - mask_3d)
    return np.clip(merged, 0, 255).astype(np.uint8)


def group_regions_by_target(regions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    target_groups: dict[str, list[dict[str, Any]]] = {}
    for region in regions:
        key = region["target"]["label"]
        target_groups.setdefault(key, []).append(region)
    return target_groups


def compute_same_target_harmony_penalty(
    regions: list[dict[str, Any]],
    candidate_map: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    penalty = 0.0
    pair_scores: dict[str, float] = {}
    for target_label, grouped_regions in group_regions_by_target(regions).items():
        if len(grouped_regions) < 2:
            continue
        for idx, region_a in enumerate(grouped_regions):
            for region_b in grouped_regions[idx + 1 :]:
                lab_a = np.asarray(candidate_map[region_a["name"]]["lab"], dtype=np.float32)
                lab_b = np.asarray(candidate_map[region_b["name"]]["lab"], dtype=np.float32)
                pair_de = float(color.deltaE_ciede2000(lab_a, lab_b))
                pair_name = f"{region_a['name']} vs {region_b['name']}"
                pair_scores[pair_name] = pair_de
                if pair_de > 0.25:
                    penalty += (pair_de - 0.25) * 1.8
    return penalty, pair_scores


def harmonize_same_target_regions(composed_img: np.ndarray, regions: list[dict[str, Any]]) -> np.ndarray:
    target_groups = [group for group in group_regions_by_target(regions).values() if len(group) > 1]
    if not target_groups:
        return composed_img
    img_f = ensure_bgr(composed_img).astype(np.float32) / 255.0
    lab_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab).astype(np.float32)
    for grouped_regions in target_groups:
        weights = []
        means = []
        for region in grouped_regions:
            mask_bool = region["mask_3d"][:, :, 0] > 0.08
            if not np.any(mask_bool):
                continue
            weights.append(float(np.count_nonzero(mask_bool)))
            means.append(np.mean(lab_f[mask_bool], axis=0))
        if len(means) < 2:
            continue
        shared_mean = np.average(np.vstack(means), axis=0, weights=np.asarray(weights, dtype=np.float32))
        for region, region_mean in zip(grouped_regions, means):
            alpha = np.clip(region["mask_3d"][:, :, 0], 0.0, 1.0).astype(np.float32)
            delta = shared_mean - region_mean
            lab_f[:, :, 0] += alpha * np.clip(delta[0] * 0.12, -1.8, 1.8)
            lab_f[:, :, 1] += alpha * np.clip(delta[1] * 0.82, -4.0, 4.0)
            lab_f[:, :, 2] += alpha * np.clip(delta[2] * 0.82, -4.0, 4.0)
    lab_f[:, :, 0] = np.clip(lab_f[:, :, 0], 0.0, 100.0)
    lab_f[:, :, 1:] = np.clip(lab_f[:, :, 1:], -127.0, 127.0)
    result = color.lab2rgb(lab_f)
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]


def build_result_combinations(orig_img: np.ndarray, regions: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
    candidate_lists = [region["candidates"][: min(3, len(region["candidates"]))] for region in regions if region["candidates"]]
    if not candidate_lists:
        return []
    if len(candidate_lists) == 1:
        combos = []
        region = regions[0]
        for cand in candidate_lists[0]:
            combos.append(
                {
                    "image": cand["image"],
                    "de": cand["de"],
                    "region_de": {region["name"]: cand["de"]},
                    "candidate_map": {region["name"]: cand},
                }
            )
        return combos[:top_n]
    combos: list[dict[str, Any]] = []
    weights = [max(1, int(np.count_nonzero(region["mask_3d"][:, :, 0] > 0.08))) for region in regions]
    for combo in product(*candidate_lists):
        composed = orig_img.copy()
        region_de: dict[str, float] = {}
        candidate_map: dict[str, Any] = {}
        for region, cand in zip(regions, combo):
            composed = composite_with_mask(composed, cand["image"], region["mask_3d"])
            region_score, _ = evaluate_delta_e(composed, region["mask_3d"], region["target"]["std_lab"])
            region_de[region["name"]] = region_score
            candidate_map[region["name"]] = cand
        harmony_penalty, harmony_pairs = compute_same_target_harmony_penalty(regions, candidate_map)
        harmonized = harmonize_same_target_regions(composed, regions) if harmony_penalty > 0.0 else composed
        if harmony_penalty > 0.0:
            adjusted_region_de: dict[str, float] = {}
            for region in regions:
                region_score, _ = evaluate_delta_e(harmonized, region["mask_3d"], region["target"]["std_lab"])
                adjusted_region_de[region["name"]] = region_score
            region_de = adjusted_region_de
            composed = harmonized
        total_de = float(np.average(list(region_de.values()), weights=weights))
        score = total_de + harmony_penalty
        combos.append(
            {
                "image": composed,
                "de": total_de,
                "score": score,
                "region_de": region_de,
                "candidate_map": candidate_map,
                "harmony_penalty": harmony_penalty,
                "harmony_pairs": harmony_pairs,
            }
        )
    unique: list[dict[str, Any]] = []
    for item in sorted(combos, key=lambda x: x.get("score", x["de"])):
        if unique and abs(item["de"] - unique[-1]["de"]) < 0.03:
            continue
        unique.append(item)
        if len(unique) >= top_n:
            break
    return unique


def image_to_base64_png(img_bgr: np.ndarray) -> str:
    png = image_to_bytes(img_bgr, ".png")
    return base64.b64encode(png).decode("ascii")


def image_to_base64_jpg(img_bgr: np.ndarray, quality: int = 96) -> str:
    jpg = image_to_bytes(img_bgr, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(jpg).decode("ascii")


def color_spec_to_dict(spec: ColorSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "hex": spec.hex,
        "rgb": list(spec.rgb),
        "hsl": list(spec.hsl),
        "lab": list(spec.lab),
        "css": spec.css,
    }


def build_result_payload(job_label: str, targets: list[dict[str, Any]], regions: list[dict[str, Any]], combos: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "job_label": job_label,
        "delta_e_method": {
            "summary": "DeltaE 使用带模特校验图的服装区域平均 CIELAB，与生成结果对应区域平均 CIELAB 做 CIEDE2000 比较。",
            "render_color_source": "渲染目标色优先来自纯色色块图；如果未上传纯色色块图，则回退为带模特校验图的服装主色。",
            "compare_color_source": "色差校验始终来自带模特校验图，保留高光和暗部影响。",
        },
        "targets": [
            {
                "label": target["label"],
                "color": color_spec_to_dict(target["spec"]),
                "style": target.get("style", "normal"),
                "render_source_kind": target.get("render_source_kind"),
                "validation_source_kind": target.get("validation_source_kind"),
                "render_source_hex": target.get("render_source_hex"),
                "reference_export_files": {
                    "validation_source": f"references/{slugify(target['label'])}_validation_source.jpg",
                    "validation_focus": f"references/{slugify(target['label'])}_validation_focus.jpg",
                    "validation_chip": f"references/{slugify(target['label'])}_validation_chip.jpg",
                    "render_source": f"references/{slugify(target['label'])}_render_source.jpg",
                    "render_chip": f"references/{slugify(target['label'])}_render_chip.jpg",
                },
            }
            for target in targets
        ],
        "regions": [
            {
                "name": region["name"],
                "target_label": region["target"]["label"],
                "best_delta_e": round(float(region["candidates"][0]["de"]), 4) if region["candidates"] else None,
                "best_params": [round(float(v), 4) for v in region["candidates"][0]["params"]] if region["candidates"] else None,
            }
            for region in regions
        ],
        "candidates": [
            {
                "rank": idx + 1,
                "delta_e": round(float(combo["de"]), 4),
                "harmony_penalty": round(float(combo.get("harmony_penalty", 0.0)), 4),
                "selection_score": round(float(combo.get("score", combo["de"])), 4),
                "same_color_pairs": {name: round(float(value), 4) for name, value in combo.get("harmony_pairs", {}).items()},
                "region_delta_e": {name: round(float(value), 4) for name, value in combo["region_de"].items()},
            }
            for idx, combo in enumerate(combos)
        ],
        "recommended_storage": {
            "ui_color": "建议保存 HEX，例如 #AABBCC，最适合 HTML / CSS / 商品系统直接调用。",
            "algorithm_color": "同时保存 CIELAB，可用于未来继续做最小 DeltaE 的自动匹配。",
        },
    }


def build_result_html(job_label: str, orig_bgr: np.ndarray, targets: list[dict[str, Any]], combos: list[dict[str, Any]]) -> str:
    sections = []
    for idx, combo in enumerate(combos, start=1):
        region_lines = "".join(
            f"<li>{name}: DeltaE {value:.2f}</li>" for name, value in combo["region_de"].items()
        )
        harmony_lines = "".join(
            f"<li>{name}: {value:.2f}</li>" for name, value in combo.get("harmony_pairs", {}).items()
        )
        compare_strip = "".join(
            [
                f"""
                <div class="compare-item">
                  <div class="compare-label">{target['label']} 校验图</div>
                  <img src="data:image/jpeg;base64,{image_to_base64_jpg(target['image_bgr'])}" />
                  <div class="compare-meta">DeltaE 校验 | {target['spec'].hex}</div>
                </div>
                <div class="compare-item">
                  <div class="compare-label">{target['label']} 渲染色</div>
                  <img src="data:image/jpeg;base64,{image_to_base64_jpg(target['render_source_bgr'])}" />
                  <div class="compare-meta">{target.get('render_source_kind', '校验模特图回退')} | {target.get('render_source_hex', target['spec'].hex)}</div>
                </div>
                """
                for target in targets
            ]
            + [
                f"""
                <div class="compare-item result">
                  <div class="compare-label">候选 {idx}</div>
                  <img src="data:image/jpeg;base64,{image_to_base64_jpg(combo['image'])}" />
                  <div class="compare-meta">DeltaE {combo['de']:.2f}</div>
                </div>
                """
            ]
        )
        sections.append(
            f"""
            <section class="card">
              <div class="head">
                <h3>候选 {idx}</h3>
                <div class="badge">总 DeltaE {combo['de']:.2f}</div>
              </div>
              <div class="compare-row">{compare_strip}</div>
              <div class="grid">
                <div>
                  <h4>区域分数</h4>
                  <ul>{region_lines}</ul>
                  <h4>同色一致性</h4>
                  <ul>{harmony_lines or '<li>当前候选没有同色区域冲突</li>'}</ul>
                </div>
                <div>
                  <h4>参考色与区域</h4>
                  <div class="target-inline">
                    {''.join(
                        f'''
                        <div class="target-inline-item">
                          <img src="data:image/jpeg;base64,{image_to_base64_jpg(target["focus_bgr"])}" />
                          <div>{target["label"]}<br/>校验图: {target["spec"].hex}<br/>渲染色: {target.get("render_source_hex", target["spec"].hex)}</div>
                        </div>
                        '''
                        for target in targets
                    )}
                  </div>
                </div>
              </div>
            </section>
            """
        )
    target_cards = "".join(
        f"""
        <div class="target">
          <div>
            <strong>{target['label']}</strong><br/>
            类型 {target.get('style', 'normal')}<br/>
            校验图颜色 {target['spec'].hex}<br/>
            渲染色来源 {target.get('render_source_kind', '校验模特图回退')}<br/>
            渲染色 {target.get('render_source_hex', target['spec'].hex)}<br/>
            RGB {target['spec'].rgb}<br/>
            HSL {target['spec'].hsl}<br/>
            LAB {target['spec'].lab}
          </div>
          <div class="target-images">
            <img src="data:image/jpeg;base64,{image_to_base64_jpg(target['image_bgr'])}" />
            <img src="data:image/jpeg;base64,{image_to_base64_jpg(target['focus_bgr'])}" />
            <img src="data:image/jpeg;base64,{image_to_base64_jpg(target['render_source_chip_bgr'])}" />
          </div>
        </div>
        """
        for target in targets
    )
    return f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <title>{job_label} 调色报告</title>
      <style>
        body {{ font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 28px; background: #f5f1ea; color: #1e293b; }}
        .hero {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 20px; align-items: start; }}
        .card {{ background: #fffdfa; border: 1px solid #dbcdb8; border-radius: 18px; padding: 18px; margin-top: 18px; box-shadow: 0 8px 24px rgba(87,65,25,0.08); }}
        .grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; }}
        .head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
        .badge {{ background:#153243; color:#f8fafc; padding:8px 12px; border-radius:999px; font-weight:700; }}
        img {{ max-width:100%; border-radius:12px; border:1px solid #e7dcc9; }}
        .target-list {{ display:grid; gap:12px; }}
        .target {{ display:grid; grid-template-columns: 0.8fr 1.2fr; gap:12px; align-items:start; background:#fff; padding:10px; border-radius:14px; border:1px solid #e7dcc9; }}
        .target-images {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; }}
        .target-images img {{ aspect-ratio: 1 / 1; object-fit: cover; }}
        .compare-row {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-bottom:16px; }}
        .compare-item {{ background:#fff; border:1px solid #e7dcc9; border-radius:14px; padding:10px; }}
        .compare-item.result {{ border-color:#153243; box-shadow: inset 0 0 0 1px #153243; }}
        .compare-label {{ font-weight:700; margin-bottom:8px; }}
        .compare-meta {{ margin-top:8px; font-size:13px; color:#475569; }}
        .target-inline {{ display:grid; gap:10px; }}
        .target-inline-item {{ display:grid; grid-template-columns: 100px 1fr; gap:10px; align-items:center; }}
      </style>
    </head>
    <body>
      <div class="hero card">
        <div>
          <h1>{job_label} 调色报告</h1>
          <p>建议未来同时保存两套颜色值：前台展示用 <strong>HEX</strong>，自动调色与最小色差匹配用 <strong>CIELAB</strong>。</p>
          <div class="target-list">{target_cards}</div>
        </div>
        <div>
          <h3>原图</h3>
          <img src="data:image/jpeg;base64,{image_to_base64_jpg(orig_bgr)}" />
        </div>
      </div>
      {''.join(sections)}
    </body>
    </html>
    """


def create_layered_psd_bytes(job_label: str, orig_bgr: np.ndarray, best_combo: dict[str, Any], targets: list[dict[str, Any]], regions: list[dict[str, Any]]) -> bytes:
    try:
        from pytoshop import enums
        from pytoshop.user import nested_layers
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"PSD 依赖不可用: {exc}") from exc

    def rgb_channels(img_bgr: np.ndarray, alpha: np.ndarray | None = None) -> dict[int, np.ndarray]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if alpha is None:
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        channels: dict[int, np.ndarray] = {
            0: rgb[:, :, 0].astype(np.uint8),
            1: rgb[:, :, 1].astype(np.uint8),
            2: rgb[:, :, 2].astype(np.uint8),
            enums.ChannelId.transparency: alpha.astype(np.uint8),
        }
        return channels

    h, w = orig_bgr.shape[:2]
    layer_stack: list[Any] = []
    reference_layers: list[Any] = []
    offset_x = 40
    for idx, target in enumerate(targets):
        swatch = create_color_chip(target["spec"].rgb, size=(200, 120), text=target["spec"].hex)
        target_group_layers = [
            nested_layers.Image(
                name=f"{target['label']}_参考原图",
                top=0,
                left=0,
                channels=rgb_channels(target["image_bgr"]),
                color_mode=enums.ColorMode.rgb,
            ),
            nested_layers.Image(
                name=f"{target['label']}_提取区域预览",
                top=0,
                left=0,
                channels=rgb_channels(target["focus_bgr"]),
                color_mode=enums.ColorMode.rgb,
            ),
            nested_layers.Image(
                name=f"{target['label']}_色卡",
                top=40 + idx * 140,
                left=offset_x,
                channels=rgb_channels(swatch),
                color_mode=enums.ColorMode.rgb,
            ),
        ]
        reference_layers.append(
            nested_layers.Group(
                name=f"{target['label']}_参考组",
                layers=target_group_layers,
                closed=False,
                visible=(idx == 0),
            )
        )
    mask_layers: list[Any] = []
    recolor_layers: list[Any] = [
        nested_layers.Image(
            name="最终合成",
            top=0,
            left=0,
            channels=rgb_channels(best_combo["image"]),
            color_mode=enums.ColorMode.rgb,
        )
    ]
    for region in regions:
        mask_alpha = np.clip(region["mask_3d"][:, :, 0] * 255.0, 0, 255).astype(np.uint8)
        mask_preview = np.dstack([mask_alpha, mask_alpha, mask_alpha])
        candidate = best_combo["candidate_map"][region["name"]]
        region_rgba = cv2.cvtColor(candidate["image"], cv2.COLOR_BGR2RGB)
        recolor_layers.append(
            nested_layers.Image(
                name=f"{region['name']}_调色层",
                top=0,
                left=0,
                channels=rgb_channels(cv2.cvtColor(region_rgba, cv2.COLOR_RGB2BGR), mask_alpha),
                color_mode=enums.ColorMode.rgb,
            )
        )
        mask_layers.append(
            nested_layers.Image(
                name=f"{region['name']}_蒙版预览",
                top=0,
                left=0,
                channels=rgb_channels(mask_preview),
                color_mode=enums.ColorMode.rgb,
            )
        )

    layer_stack.extend(
        [
            nested_layers.Group(name="参考色层", layers=reference_layers, closed=False),
            nested_layers.Group(name="蒙版层", layers=mask_layers, visible=False, closed=False),
            nested_layers.Group(name="调色结果层", layers=recolor_layers, closed=False),
            nested_layers.Image(
                name="原始底图",
                top=0,
                left=0,
                channels=rgb_channels(orig_bgr),
                color_mode=enums.ColorMode.rgb,
            ),
        ]
    )

    psd = nested_layers.nested_layers_to_psd(
        layer_stack,
        color_mode=enums.ColorMode.rgb,
        size=(h, w),
        compression=enums.Compression.raw,
    )
    buffer = io.BytesIO()
    psd.write(buffer)
    return buffer.getvalue()


def save_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def cleanup_legacy_pngs(folder: Path) -> None:
    if not folder.exists():
        return
    for path in folder.rglob("*.png"):
        try:
            path.unlink()
        except OSError:
            continue


def build_export_zip(result: dict[str, Any]) -> bytes:
    if not result["combos"]:
        return b""
    best = result["combos"][0]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        base = slugify(result["job_label"])
        zf.writestr(
            f"{base}/best.jpg",
            image_to_bytes(best["image"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 100]),
        )
        zf.writestr(f"{base}/best.psd", result["psd_bytes"])
        zf.writestr(f"{base}/report.html", result["html"].encode("utf-8"))
        zf.writestr(
            f"{base}/report.json",
            json.dumps(result["payload"], ensure_ascii=False, indent=2).encode("utf-8"),
        )
        for idx, target in enumerate(result["targets"], start=1):
            prefix = f"{base}/references/{idx:02d}_{slugify(target['label'])}"
            zf.writestr(f"{prefix}_source.jpg", image_to_bytes(target["image_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 97]))
            zf.writestr(f"{prefix}_focus.jpg", image_to_bytes(target["focus_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 97]))
            zf.writestr(f"{prefix}_chip.jpg", image_to_bytes(target["chip_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 97]))
    return buffer.getvalue()


def discover_sample_bundle(sample_name: str) -> dict[str, Any]:
    folder = APP_DIR / sample_name
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Sample folder not found: {folder}")
    files = sorted([path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES])
    if not files:
        raise FileNotFoundError(f"No image files found in sample folder: {folder}")
    if sample_name in {"A", "B"}:
        orig_path = next(path for path in files if ("上衣" not in path.name and "底裤" not in path.name))
        top_path = next(path for path in files if "上衣" in path.name)
        bottom_path = next(path for path in files if "底裤" in path.name)
        return {
            "label": sample_name,
            "region_count": 2,
            "orig_img": read_image_path(orig_path),
            "region_sources": [
                {"name": "上衣", "mask_source": read_image_path(top_path)},
                {"name": "底裤", "mask_source": read_image_path(bottom_path)},
            ],
        }
    if sample_name == "E":
        orig_path = folder / "11.jpg"
        mask_path = folder / "11.png"
        if not orig_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Sample E is missing required files: {folder}")
        return {
            "label": sample_name,
            "region_count": 1,
            "orig_img": read_image_path(orig_path),
            "region_sources": [{"name": "??", "mask_source": read_image_path(mask_path)}],
        }
    alpha_files = [path for path in files if read_image_path(path) is not None and extract_alpha(read_image_path(path)) is not None]
    mask_path = alpha_files[0] if alpha_files else files[-1]
    orig_path = next(path for path in files if path != mask_path)
    return {
        "label": sample_name,
        "region_count": 1,
        "orig_img": read_image_path(orig_path),
        "region_sources": [{"name": "主体", "mask_source": read_image_path(mask_path)}],
    }


def available_sample_names() -> list[str]:
    names: list[str] = []
    for sample_name in ["A", "B", "C", "D", "E"]:
        folder = APP_DIR / sample_name
        if not folder.exists() or not folder.is_dir():
            continue
        try:
            has_images = any(path.suffix.lower() in IMAGE_SUFFIXES for path in folder.iterdir())
        except OSError:
            has_images = False
        if has_images:
            names.append(sample_name)
    return names


def list_reference_paths() -> list[Path]:
    folder = APP_DIR / "颜色参考"
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES])


def select_reference_paths_for_styles() -> dict[str, Path]:
    selected: dict[str, Path] = {}
    fallback: list[Path] = []
    for path in list_reference_paths():
        img = read_image_path(path)
        if img is None:
            continue
        analysis = analyze_validation_reference_image(img, path.stem)
        style = analysis.get("style", "normal")
        fallback.append(path)
        if style not in selected:
            selected[style] = path
    if "white" not in selected and "light" in selected:
        selected["white"] = selected["light"]
    if "light" not in selected and "white" in selected:
        selected["light"] = selected["white"]
    if "dark" not in selected and fallback:
        selected["dark"] = fallback[-1]
    if "neon" not in selected and fallback:
        selected["neon"] = fallback[min(1, len(fallback) - 1)]
    if "normal" not in selected and fallback:
        selected["normal"] = fallback[0]
    return selected


def build_job_inputs(
    job_label: str,
    orig_img: np.ndarray,
    region_sources: list[dict[str, Any]],
    ref_inputs: list[dict[str, Any]],
    region_to_target: list[int],
    top_n: int = 3,
) -> dict[str, Any]:
    orig_bgr = ensure_bgr(orig_img)
    targets = [analyze_target_input(item) for item in ref_inputs]
    regions = []
    for index, region in enumerate(region_sources):
        mask_3d = preprocess_mask(region["mask_source"], orig_bgr.shape[:2])
        candidates = optimize_region_candidates(orig_bgr, mask_3d, targets[region_to_target[index]], region["name"], top_n=max(top_n + 1, 4))
        regions.append(
            {
                "name": region["name"],
                "mask_3d": mask_3d,
                "target": targets[region_to_target[index]],
                "candidates": candidates,
            }
        )
    combos = build_result_combinations(orig_bgr, regions, top_n=top_n)
    best_combo = combos[0] if combos else None
    payload = build_result_payload(job_label, targets, regions, combos)
    html = build_result_html(job_label, orig_bgr, targets, combos) if combos else ""
    psd_bytes = create_layered_psd_bytes(job_label, orig_bgr, best_combo, targets, regions) if best_combo else b""
    return {
        "job_label": job_label,
        "orig_bgr": orig_bgr,
        "targets": targets,
        "regions": regions,
        "combos": combos,
        "payload": payload,
        "html": html,
        "psd_bytes": psd_bytes,
    }


def analyze_reference_folder(reference_paths: list[Path] | None = None) -> dict[str, Any]:
    paths = reference_paths or list_reference_paths()
    rows = []
    for path in paths:
        img = read_image_path(path)
        if img is None:
            continue
        analysis = analyze_reference_image(img, path.stem)
        rows.append(
            {
                "file_name": path.name,
                "hex": analysis["spec"].hex,
                "rgb": list(analysis["spec"].rgb),
                "hsl": list(analysis["spec"].hsl),
                "lab": list(analysis["spec"].lab),
                "css": analysis["spec"].css,
                "style": analysis["style"],
                "chip_jpg": image_to_base64_jpg(analysis["chip_bgr"]),
                "source_jpg": image_to_base64_jpg(analysis["image_bgr"]),
                "focus_jpg": image_to_base64_jpg(analysis["focus_bgr"]),
            }
        )
    html_rows = "".join(
        f"""
        <tr>
          <td>{row['file_name']}</td>
          <td>
            <div style="display:grid;grid-template-columns:repeat(3,minmax(120px,1fr));gap:10px;align-items:start;">
              <figure style="margin:0;">
                <img src="data:image/jpeg;base64,{row['source_jpg']}" style="width:100%;border-radius:10px;border:1px solid #d8cdbd;" />
                <figcaption style="margin-top:6px;font-size:12px;color:#475569;">原始模特图</figcaption>
              </figure>
              <figure style="margin:0;">
                <img src="data:image/jpeg;base64,{row['focus_jpg']}" style="width:100%;border-radius:10px;border:1px solid #d8cdbd;" />
                <figcaption style="margin-top:6px;font-size:12px;color:#475569;">提取区域预览</figcaption>
              </figure>
              <figure style="margin:0;">
                <img src="data:image/jpeg;base64,{row['chip_jpg']}" style="width:100%;border-radius:10px;border:1px solid #d8cdbd;" />
                <figcaption style="margin-top:6px;font-size:12px;color:#475569;">最终色卡</figcaption>
              </figure>
            </div>
          </td>
          <td>{row['hex']}</td>
          <td>{tuple(row['rgb'])}</td>
          <td>{tuple(row['hsl'])}</td>
          <td>{tuple(round(v, 2) for v in row['lab'])}</td>
          <td>{row['style']}</td>
          <td><code>{row['css']}</code></td>
        </tr>
        """
        for row in rows
    )
    html = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <title>颜色参考目录分析</title>
      <style>
        body {{ font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 24px; background: #f7f3ed; color: #1f2937; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 16px; overflow: hidden; }}
        th, td {{ padding: 12px 14px; border-bottom: 1px solid #eadfce; text-align: left; vertical-align: top; }}
        th {{ background: #163547; color: #f8fafc; }}
        code {{ white-space: normal; }}
      </style>
    </head>
    <body>
      <h1>颜色参考目录分析</h1>
      <p>建议未来把颜色同时保存成 <strong>HEX</strong> 和 <strong>CIELAB</strong>。HEX 适合 HTML / CSS / 前台展示，LAB 适合后续自动调色与最小色差匹配。</p>
      <table>
        <thead>
          <tr>
            <th>文件</th>
            <th>对比图</th>
            <th>HEX</th>
            <th>RGB</th>
            <th>HSL</th>
            <th>LAB</th>
            <th>分类</th>
            <th>推荐 CSS</th>
          </tr>
        </thead>
        <tbody>{html_rows}</tbody>
      </table>
    </body>
    </html>
    """
    return {"rows": rows, "html": html, "json": json.dumps(rows, ensure_ascii=False, indent=2)}


def run_demo_batch_tests(max_cases: int = 6) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not available_sample_names():
        raise ValueError("No local sample folders available for batch testing.")
    style_refs = select_reference_paths_for_styles()
    required_styles = {"white", "light", "dark", "neon"}
    if not required_styles.issubset(style_refs.keys()):
        raise ValueError("Reference images are missing for one or more required style categories.")
    cases = [
        ("C_白色_一件套", "C", [style_refs["white"]], [0], ["white"]),
        ("C_深色_一件套", "C", [style_refs["dark"]], [0], ["dark"]),
        ("A_深色_同色", "A", [style_refs["dark"]], [0, 0], ["dark"]),
        ("A_荧光_浅色_异色", "A", [style_refs["neon"], style_refs["light"]], [0, 1], ["neon", "light"]),
        ("B_浅色_同色", "B", [style_refs["light"]], [0, 0], ["light"]),
        ("B_深色_荧光_异色", "B", [style_refs["dark"], style_refs["neon"]], [0, 1], ["dark", "neon"]),
    ][:max_cases]
    summaries = []
    report_rows = []
    for label, sample_name, ref_selection, region_map, styles in cases:
        sample = discover_sample_bundle(sample_name)
        ref_inputs = [{"label": path.stem, "image": read_image_path(path)} for path in ref_selection]
        result = build_job_inputs(label, sample["orig_img"], sample["region_sources"], ref_inputs, region_map, top_n=3)
        case_dir = OUTPUT_DIR / slugify(label)
        cleanup_legacy_pngs(case_dir)
        best = result["combos"][0]
        jpg_path = save_bytes(case_dir / "best.jpg", image_to_bytes(best["image"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 100]))
        psd_path = save_bytes(case_dir / "best.psd", result["psd_bytes"])
        json_path = save_bytes(case_dir / "report.json", json.dumps(result["payload"], ensure_ascii=False, indent=2).encode("utf-8"))
        html_path = save_bytes(case_dir / "report.html", result["html"].encode("utf-8"))
        zip_path = save_bytes(case_dir / "export_bundle.zip", build_export_zip(result))
        refs_dir = case_dir / "references"
        for idx, target in enumerate(result["targets"], start=1):
            prefix = refs_dir / f"{idx:02d}_{slugify(target['label'])}"
            save_bytes(prefix.with_name(prefix.name + "_source.jpg"), image_to_bytes(target["image_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 97]))
            save_bytes(prefix.with_name(prefix.name + "_focus.jpg"), image_to_bytes(target["focus_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 97]))
            save_bytes(prefix.with_name(prefix.name + "_chip.jpg"), image_to_bytes(target["chip_bgr"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 97]))
        summaries.append(
            {
                "label": label,
                "styles": ", ".join(styles),
                "best_de": round(float(best["de"]), 4),
                "jpg_path": str(jpg_path),
                "psd_path": str(psd_path),
                "zip_path": str(zip_path),
                "json_path": str(json_path),
                "html_path": str(html_path),
            }
        )
        report_rows.append(
            f"""
            <section class="card">
              <h2>{label}</h2>
              <p><strong>测试分类:</strong> {", ".join(styles)}</p>
              <p><strong>最佳 DeltaE:</strong> {best['de']:.2f}</p>
              <p>
                <a href="{jpg_path.name}">JPG</a> |
                <a href="{psd_path.name}">PSD</a> |
                <a href="{zip_path.name}">导出包 ZIP</a> |
                <a href="{json_path.name}">JSON</a>
              </p>
              <img src="data:image/jpeg;base64,{image_to_base64_jpg(best['image'])}" />
            </section>
            """
        )
        gc.collect()
    palette_report = analyze_reference_folder()
    palette_dir = OUTPUT_DIR / "reference_palette"
    cleanup_legacy_pngs(palette_dir)
    palette_html_path = save_bytes(palette_dir / "palette.html", palette_report["html"].encode("utf-8"))
    palette_json_path = save_bytes(palette_dir / "palette.json", palette_report["json"].encode("utf-8"))
    summary_html = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <title>批量调色测试报告</title>
      <style>
        body {{ font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 24px; background: #f3efe7; color: #1e293b; }}
        .card {{ background: white; border-radius: 18px; padding: 18px; border: 1px solid #e4d6c4; margin-bottom: 18px; }}
        img {{ max-width: 100%; border-radius: 12px; border: 1px solid #eadfce; }}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>批量调色测试报告</h1>
        <p>这里已经连续跑了多组一件套 / 两件套 / 同色 / 异色场景，并输出最小 DeltaE 的结果图和 PSD。</p>
        <p><a href="reference_palette/palette.html">颜色参考目录 HTML</a> | <a href="reference_palette/palette.json">颜色参考目录 JSON</a></p>
      </div>
      {''.join(report_rows)}
    </body>
    </html>
    """
    summary_path = save_bytes(OUTPUT_DIR / "batch_test_summary.html", summary_html.encode("utf-8"))
    return {
        "cases": summaries,
        "report_path": str(summary_path),
        "palette_html_path": str(palette_html_path),
        "palette_json_path": str(palette_json_path),
    }


def render_color_summary(targets: list[dict[str, Any]]) -> None:
    cols = st.columns(max(1, len(targets)))
    for col, target in zip(cols, targets):
        with col:
            st.image(cv2.cvtColor(target["chip_bgr"], cv2.COLOR_BGR2RGB), caption=target["label"], use_container_width=True)
            st.code(
                "\n".join(
                    [
                        f"HEX: {target['spec'].hex}",
                        f"RGB: {target['spec'].rgb}",
                        f"HSL: {target['spec'].hsl}",
                        f"LAB: {target['spec'].lab}",
                    ]
                ),
                language="text",
            )


def render_result_downloads(result: dict[str, Any]) -> None:
    if not result["combos"]:
        st.warning("????????????????????")
        return

    best = result["combos"][0]
    jpg_bytes = image_to_bytes(best["image"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    json_bytes = json.dumps(result["payload"], ensure_ascii=False, indent=2).encode("utf-8")
    export_state_key = f"stable_advanced_exports::{result['job_label']}"
    export_state = st.session_state.get(export_state_key)

    cols = st.columns(5)
    with cols[0]:
        st.download_button("???? JPG", jpg_bytes, file_name=f"{slugify(result['job_label'])}_best.jpg", mime="image/jpeg", use_container_width=True)
    with cols[1]:
        if export_state:
            st.download_button("???? PSD", export_state["psd_bytes"], file_name=f"{slugify(result['job_label'])}_best.psd", mime="image/vnd.adobe.photoshop", use_container_width=True)
        else:
            st.button("???? PSD", disabled=True, use_container_width=True, key=f"disabled_psd_{slugify(result['job_label'])}")
    with cols[2]:
        if export_state:
            st.download_button("????? ZIP", export_state["zip_bytes"], file_name=f"{slugify(result['job_label'])}_export.zip", mime="application/zip", use_container_width=True)
        else:
            st.button("????? ZIP", disabled=True, use_container_width=True, key=f"disabled_zip_{slugify(result['job_label'])}")
    with cols[3]:
        if export_state:
            st.download_button("???? HTML", export_state["html_bytes"], file_name=f"{slugify(result['job_label'])}_report.html", mime="text/html", use_container_width=True)
        else:
            st.button("???? HTML", disabled=True, use_container_width=True, key=f"disabled_html_{slugify(result['job_label'])}")
    with cols[4]:
        st.download_button("???? JSON", json_bytes, file_name=f"{slugify(result['job_label'])}_report.json", mime="application/json", use_container_width=True)

    with st.expander("???????PSD / ZIP / HTML?", expanded=export_state is None):
        if export_state is None:
            if st.button("??????", key=f"prepare_advanced_exports_{slugify(result['job_label'])}", use_container_width=True):
                with st.spinner("???????????????????????..."):
                    html = build_result_html(result["job_label"], result["orig_bgr"], result["targets"], result["combos"])
                    psd_bytes = create_layered_psd_bytes(result["job_label"], result["orig_bgr"], result["combos"][0], result["targets"], result["regions"])
                    advanced_result = {**result, "html": html, "psd_bytes": psd_bytes}
                    zip_bytes = build_export_zip(advanced_result)
                st.session_state[export_state_key] = {
                    "html_bytes": html.encode("utf-8"),
                    "psd_bytes": psd_bytes,
                    "zip_bytes": zip_bytes,
                }
                st.rerun()
        else:
            st.success("???????????????????")

def render_candidate_gallery(result: dict[str, Any]) -> None:
    combos = result["combos"][:STREAMLIT_SAFE_TOP_N]
    if not combos:
        return
    st.markdown("**?????**")
    ref_cols = st.columns(max(1, len(result["targets"])))
    for idx, target in enumerate(result["targets"]):
        with ref_cols[idx]:
            st.image(cv2.cvtColor(thumbnail_for_ui(target["image_bgr"], 170, 190), cv2.COLOR_BGR2RGB), caption=target["label"], use_container_width=False)
    st.markdown("**????**")
    cols = st.columns(max(1, len(combos)))
    for idx, combo in enumerate(combos):
        with cols[idx]:
            st.image(cv2.cvtColor(thumbnail_for_ui(combo["image"], 180, 230), cv2.COLOR_BGR2RGB), caption=f"?? {idx + 1}", use_container_width=False)
            st.caption(f"DeltaE {combo['de']:.2f}")

def build_single_job_ui() -> None:
    result_state_key = "stable_last_result"
    sample_names = available_sample_names()
    has_local_samples = bool(sample_names)
    top_cols = st.columns([1.1, 1.1, 1.0, 1.0])
    source_options = ["????", "????"] if has_local_samples else ["????"]
    with top_cols[0]:
        source_mode = st.radio("????", source_options, horizontal=True, label_visibility="collapsed")

    sample_name = "manual"
    region_count = 1
    orig_img = None
    region_sources: list[dict[str, Any]] = []
    if source_mode == "????" and has_local_samples:
        with top_cols[1]:
            sample_name = st.selectbox("????", sample_names, label_visibility="collapsed")
        try:
            sample = discover_sample_bundle(sample_name)
        except FileNotFoundError:
            st.warning("?????????????????????")
            source_mode = "????"
            sample = None
        if sample is not None:
            region_count = sample["region_count"]
            orig_img = constrain_image_for_streamlit(sample["orig_img"])
            region_sources = [{"name": item["name"], "mask_source": constrain_image_for_streamlit(item["mask_source"])} for item in sample["region_sources"]]
        else:
            with top_cols[2]:
                region_count = st.radio("????", [1, 2], horizontal=True, format_func=lambda value: "???" if value == 1 else "???", label_visibility="collapsed")
            orig_cols = st.columns(3 if region_count == 1 else 4)
            with orig_cols[0]:
                orig_file = st.file_uploader("??", type=["jpg", "jpeg", "png"])
            orig_img = load_uploaded_image(orig_file)
            if region_count == 1:
                with orig_cols[1]:
                    mask_file = st.file_uploader("????", type=["jpg", "jpeg", "png"], key="stable_mask_one_fallback")
                region_sources.append({"name": "??", "mask_source": load_uploaded_image(mask_file)})
            else:
                with orig_cols[1]:
                    top_file = st.file_uploader("????", type=["jpg", "jpeg", "png"], key="stable_mask_top_fallback")
                with orig_cols[2]:
                    bottom_file = st.file_uploader("????", type=["jpg", "jpeg", "png"], key="stable_mask_bottom_fallback")
                region_sources.extend([
                    {"name": "??", "mask_source": load_uploaded_image(top_file)},
                    {"name": "??", "mask_source": load_uploaded_image(bottom_file)},
                ])
    else:
        with top_cols[1]:
            region_count = st.radio("????", [1, 2], horizontal=True, format_func=lambda value: "???" if value == 1 else "???", label_visibility="collapsed")
        orig_cols = st.columns(3 if region_count == 1 else 4)
        with orig_cols[0]:
            orig_file = st.file_uploader("??", type=["jpg", "jpeg", "png"])
        orig_img = load_uploaded_image(orig_file)
        if region_count == 1:
            with orig_cols[1]:
                mask_file = st.file_uploader("????", type=["jpg", "jpeg", "png"], key="stable_mask_one")
            region_sources.append({"name": "??", "mask_source": load_uploaded_image(mask_file)})
        else:
            with orig_cols[1]:
                top_file = st.file_uploader("????", type=["jpg", "jpeg", "png"], key="stable_mask_top")
            with orig_cols[2]:
                bottom_file = st.file_uploader("????", type=["jpg", "jpeg", "png"], key="stable_mask_bottom")
            region_sources.extend([
                {"name": "??", "mask_source": load_uploaded_image(top_file)},
                {"name": "??", "mask_source": load_uploaded_image(bottom_file)},
            ])

    color_count = 1
    if region_count != 1:
        with top_cols[2]:
            color_count = st.radio("????", [1, 2], horizontal=True, format_func=lambda value: "????" if value == 1 else "????", label_visibility="collapsed")
    ref_paths = list_reference_paths() if source_mode == "????" else []
    ref_inputs: list[dict[str, Any]] = []
    ref_name_map = {path.name: path for path in ref_paths}
    ref_name_options = list(ref_name_map.keys())
    if source_mode == "????" and not ref_paths:
        source_mode = "????"
    ref_cols = st.columns(color_count if color_count > 0 else 1)
    for idx in range(color_count):
        label_default = f"?? {idx + 1}"
        with ref_cols[idx]:
            st.markdown(f"**{label_default}**")
            if source_mode == "????" and ref_name_options:
                default_index = min(idx, len(ref_name_options) - 1)
                validation_name = st.selectbox(f"{label_default} ???", ref_name_options, index=default_index, key=f"stable_validation_ref_{idx}")
                validation_path = ref_name_map[validation_name]
                validation_image = constrain_image_for_streamlit(read_image_path(validation_path))
                label = validation_path.stem
            else:
                validation_file = st.file_uploader(f"{label_default} ???", type=["jpg", "jpeg", "png"], key=f"stable_validation_upload_{idx}")
                validation_image = load_uploaded_image(validation_file)
                label = Path(validation_file.name).stem if validation_file is not None else label_default
            render_file = st.file_uploader(f"{label_default} ???", type=["jpg", "jpeg", "png"], key=f"stable_render_upload_{idx}")
            render_image = load_uploaded_image(render_file)
            ref_inputs.append({"label": label, "validation_image": validation_image, "render_image": render_image})

    region_map = [0] if region_count == 1 else ([0, 0] if color_count == 1 else [0, 1])
    if st.button("???????????", use_container_width=True):
        if orig_img is None:
            st.error("???????")
            return
        if any(item["mask_source"] is None for item in region_sources):
            st.error("?????????????????")
            return
        if any(item["validation_image"] is None for item in ref_inputs):
            st.error("????????????????")
            return
        st.info("????????????????????????????????")
        with st.spinner("???????????..."):
            result = build_job_inputs(
                "??????" if source_mode == "????" else f"{sample_name}_????",
                orig_img,
                region_sources,
                ref_inputs,
                region_map,
                top_n=STREAMLIT_SAFE_TOP_N,
            )
        st.session_state[result_state_key] = result
    result = st.session_state.get(result_state_key)
    if result and result.get("combos"):
        st.success(f"??????? {len(result['combos'])} ??????")
        render_result_downloads(result)
        render_candidate_gallery(result)

def main() -> None:
    st.set_page_config(page_title="???????? - ?????", layout="wide")
    inject_css()
    st.title("???????? - ?????")
    st.caption("???????? Streamlit Cloud ???????????????????????????")
    build_single_job_ui()

if __name__ == "__main__":
    main()
