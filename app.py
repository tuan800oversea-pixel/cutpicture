from io import BytesIO
from itertools import permutations
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np
from PIL import Image
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
TEST_ROOT = BASE_DIR / "test_images"
TEST_ORG_DIR = TEST_ROOT / "originals"
TEST_REF_DIR = TEST_ROOT / "templates"
TEST_OUTPUT_DIR = TEST_ROOT / "output"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_raw_image(uploaded_file):
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def load_path_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def convert_cv_to_bytes(cv_img, output_format="JPEG"):
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img_rgb)
    buf = BytesIO()

    if output_format.upper() == "PNG":
        pil_img.save(buf, format="PNG", compress_level=1, dpi=(300, 300))
        extension = "png"
    else:
        pil_img.save(
            buf,
            format="JPEG",
            quality=98,
            subsampling=0,
            optimize=True,
            dpi=(300, 300),
        )
        extension = "jpg"

    return buf.getvalue(), extension


def natural_sort_key(value):
    stem = value.stem if isinstance(value, Path) else Path(value).stem
    parts = re.findall(r"\d+|\D+", stem.lower())
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def get_item_name(item):
    return item.name if not isinstance(item, Path) else item.name


def find_matching_reference(org_filename, ref_files):
    org_base = Path(org_filename).stem.lower()
    sorted_refs = sorted(ref_files, key=lambda item: natural_sort_key(get_item_name(item)))

    exact_match = None
    suffix_match = None
    for ref in sorted_refs:
        ref_base = Path(get_item_name(ref)).stem.lower()
        if org_base == ref_base:
            exact_match = ref
            break
        if org_base.endswith(ref_base):
            suffix_match = ref

    return exact_match or suffix_match


def build_file_pairs(org_files, ref_files):
    remaining_refs = list(ref_files)
    pairs = []
    pairing_mode = "文件名自动匹配"

    matched_orgs = set()
    for org_file in org_files:
        matched_ref = find_matching_reference(get_item_name(org_file), remaining_refs)
        if matched_ref is None:
            continue
        pairs.append((org_file, matched_ref))
        matched_orgs.add(org_file)
        remaining_refs.remove(matched_ref)

    unmatched_orgs = [item for item in org_files if item not in matched_orgs]
    if unmatched_orgs and remaining_refs and len(unmatched_orgs) == len(remaining_refs):
        pairing_mode = "按排序顺序补全配对"
        sorted_orgs = sorted(unmatched_orgs, key=lambda item: natural_sort_key(get_item_name(item)))
        sorted_refs = sorted(remaining_refs, key=lambda item: natural_sort_key(get_item_name(item)))
        pairs.extend(zip(sorted_orgs, sorted_refs))

    return sorted(pairs, key=lambda pair: natural_sort_key(get_item_name(pair[0]))), pairing_mode


def build_reference_triples(org_files, cropped_ref_files, uncut_ref_files=None):
    cropped_pairs, cropped_mode = build_file_pairs(org_files, cropped_ref_files)
    uncut_map = {}
    uncut_mode = ""

    if uncut_ref_files:
        uncut_pairs, uncut_mode = build_file_pairs(org_files, uncut_ref_files)
        uncut_map = {org: ref for org, ref in uncut_pairs}

    triples = []
    for org_file, cropped_ref in cropped_pairs:
        triples.append(
            {
                "original": org_file,
                "cropped_ref": cropped_ref,
                "uncut_ref": uncut_map.get(org_file),
            }
        )

    return triples, cropped_mode, uncut_mode


def split_local_reference_sets(org_files, ref_files):
    org_stems = {path.stem.lower() for path in org_files}
    uncut_refs = [path for path in ref_files if path.stem.lower() in org_stems]
    cropped_refs = [path for path in ref_files if path not in uncut_refs]
    return cropped_refs, uncut_refs


def list_image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES],
        key=natural_sort_key,
    )


def discover_local_groups():
    if not TEST_ORG_DIR.exists() or not TEST_REF_DIR.exists():
        return []

    org_groups = {path.name for path in TEST_ORG_DIR.iterdir() if path.is_dir()}
    ref_groups = {path.name for path in TEST_REF_DIR.iterdir() if path.is_dir()}
    return sorted(org_groups & ref_groups, key=natural_sort_key)


def largest_component_mask(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def cleanup_mask(mask, kernel_size=5, close_iterations=2, dilate_iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    mask = largest_component_mask(mask)
    if cv2.countNonZero(mask) == 0:
        return mask
    if dilate_iterations > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    return mask


def resize_with_max_side(img, max_side=1400, interpolation=cv2.INTER_AREA):
    height, width = img.shape[:2]
    scale_down = max(height, width) / max_side if max(height, width) > max_side else 1.0
    if scale_down == 1.0:
        return img.copy(), scale_down

    new_w = max(1, int(round(width / scale_down)))
    new_h = max(1, int(round(height / scale_down)))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation), scale_down


def build_white_background_mask(img):
    diff_from_white = 255 - img.astype(np.int16)
    max_diff = diff_from_white.max(axis=2).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.where((max_diff > 16) | (gray < 240), 255, 0).astype(np.uint8)
    return cleanup_mask(mask, kernel_size=5, close_iterations=2, dilate_iterations=1)


def mask_area_ratio(mask):
    height, width = mask.shape[:2]
    return cv2.countNonZero(mask) / max(height * width, 1)


def mask_border_touch_ratio(mask):
    top = mask[0, :]
    bottom = mask[-1, :]
    left = mask[:, 0]
    right = mask[:, -1]
    border = np.concatenate([top, bottom, left, right])
    return np.count_nonzero(border) / max(border.size, 1)


def mask_is_degenerate(mask):
    area_ratio = mask_area_ratio(mask)
    border_ratio = mask_border_touch_ratio(mask)
    return area_ratio < 0.01 or area_ratio > 0.96 or border_ratio > 0.97


def build_grabcut_mask(img):
    small_img, scale_down = resize_with_max_side(img, max_side=1200, interpolation=cv2.INTER_AREA)
    height, width = small_img.shape[:2]
    if min(height, width) < 80:
        return None

    rect = (
        int(width * 0.08),
        int(height * 0.02),
        max(1, int(width * 0.84)),
        max(1, int(height * 0.96)),
    )
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small_img, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg_mask = cleanup_mask(fg_mask, kernel_size=7, close_iterations=2, dilate_iterations=1)
    if cv2.countNonZero(fg_mask) == 0:
        return None

    if scale_down != 1.0:
        fg_mask = cv2.resize(fg_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return fg_mask


def extract_foreground_mask(img, allow_grabcut=True):
    if img is None:
        return None

    simple_mask = build_white_background_mask(img)
    if not allow_grabcut or not mask_is_degenerate(simple_mask):
        if cv2.countNonZero(simple_mask) > 0:
            return simple_mask

    if allow_grabcut:
        grabcut_mask = build_grabcut_mask(img)
        if grabcut_mask is not None and cv2.countNonZero(grabcut_mask) > 0:
            return grabcut_mask

    if cv2.countNonZero(simple_mask) > 0:
        return simple_mask

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.full(gray.shape, 255, dtype=np.uint8)


def get_mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        height, width = mask.shape[:2]
        return 0.0, 0.0, float(width), float(height)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)


def get_mask_centroid(mask):
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] > 0:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    x, y, width, height = get_mask_bbox(mask)
    return x + width / 2.0, y + height / 2.0


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_right = ax + aw
    a_bottom = ay + ah
    b_right = bx + bw
    b_bottom = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_right, b_right)
    inter_y2 = min(a_bottom, b_bottom)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def score_mask_alignment(mask_a, mask_b):
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    mask_iou = inter / max(union, 1)

    area_a = cv2.countNonZero(mask_a)
    area_b = cv2.countNonZero(mask_b)
    area_ratio = min(area_a, area_b) / max(area_a, area_b, 1)
    bbox_score = bbox_iou(get_mask_bbox(mask_a), get_mask_bbox(mask_b))

    ax, ay, aw, ah = get_mask_bbox(mask_a)
    bx, by, bw, bh = get_mask_bbox(mask_b)
    top_score = 1.0 - min(abs(ay - by) / max(mask_b.shape[0], 1), 1.0)
    bottom_score = 1.0 - min(abs((ay + ah) - (by + bh)) / max(mask_b.shape[0], 1), 1.0)

    cx_a, cy_a = get_mask_centroid(mask_a)
    cx_b, cy_b = get_mask_centroid(mask_b)
    height, width = mask_b.shape[:2]
    center_dist = np.hypot(cx_a - cx_b, cy_a - cy_b)
    diag = np.hypot(width, height)
    center_score = 1.0 - min(center_dist / max(diag, 1.0), 1.0)

    return (
        0.48 * mask_iou
        + 0.14 * bbox_score
        + 0.12 * area_ratio
        + 0.10 * center_score
        + 0.10 * top_score
        + 0.06 * bottom_score
    )


def resize_image_and_mask(img, mask, max_size=1200):
    height, width = img.shape[:2]
    scale_down = max(height, width) / max_size if max(height, width) > max_size else 1.0

    if scale_down == 1.0:
        return img.copy(), mask.copy(), scale_down

    new_w = max(1, int(round(width / scale_down)))
    new_h = max(1, int(round(height / scale_down)))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized_img, resized_mask, scale_down


def affine_to_homogeneous(matrix_2x3):
    return np.vstack([matrix_2x3, [0.0, 0.0, 1.0]]).astype(np.float32)


def homogeneous_to_affine(matrix_3x3):
    return matrix_3x3[:2].astype(np.float32)


def compose_affine(matrix_a, matrix_b):
    return homogeneous_to_affine(affine_to_homogeneous(matrix_a) @ affine_to_homogeneous(matrix_b))


def scale_affine_from_small_to_large(matrix_small, scale_down_org, scale_down_ref):
    homo = affine_to_homogeneous(matrix_small)
    scale_ref = np.array(
        [[scale_down_ref, 0.0, 0.0], [0.0, scale_down_ref, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    scale_org_inv = np.array(
        [[1.0 / scale_down_org, 0.0, 0.0], [0.0, 1.0 / scale_down_org, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return homogeneous_to_affine(scale_ref @ homo @ scale_org_inv)


def estimate_initial_transform_from_masks(org_mask, ref_mask):
    _, _, org_w, org_h = get_mask_bbox(org_mask)
    _, _, ref_w, ref_h = get_mask_bbox(ref_mask)
    org_area = max(float(cv2.countNonZero(org_mask)), 1.0)
    ref_area = max(float(cv2.countNonZero(ref_mask)), 1.0)

    scale_candidates = [
        ref_w / max(org_w, 1.0),
        ref_h / max(org_h, 1.0),
        np.sqrt(ref_area / org_area),
    ]
    scale = float(np.median(scale_candidates))
    scale = float(np.clip(scale, 0.2, 6.0))

    org_cx, org_cy = get_mask_centroid(org_mask)
    ref_cx, ref_cy = get_mask_centroid(ref_mask)
    tx = ref_cx - scale * org_cx
    ty = ref_cy - scale * org_cy
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)


def build_sift_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.GaussianBlur(enhanced, (0, 0), 0.6)


def build_ecc_map(img, mask):
    gray = build_sift_image(img).astype(np.float32) / 255.0
    edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 120).astype(np.float32) / 255.0

    if cv2.countNonZero(mask) > 0:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    else:
        dist = np.zeros_like(gray, dtype=np.float32)

    combined = 0.45 * dist + 0.35 * gray + 0.20 * edges
    combined[mask == 0] = 0.0
    return combined.astype(np.float32)


def align_scene_full_frame_fast(org_img_highres, ref_img_highres):
    org_small, scale_down_org = resize_with_max_side(org_img_highres, max_side=1000, interpolation=cv2.INTER_AREA)
    ref_small, scale_down_ref = resize_with_max_side(ref_img_highres, max_side=1000, interpolation=cv2.INTER_AREA)
    org_gray = build_sift_image(org_small).astype(np.float32) / 255.0
    ref_gray = build_sift_image(ref_small).astype(np.float32) / 255.0

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)

    try:
        ecc_score, warp_matrix = cv2.findTransformECC(
            ref_gray,
            org_gray,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return None, [], 0.0

    matrix_highres = scale_affine_from_small_to_large(
        warp_matrix,
        scale_down_org,
        scale_down_ref,
    )
    if not is_transform_reasonable(matrix_highres, ref_img_highres.shape):
        return None, [], 0.0

    aligned_highres = cv2.warpAffine(
        org_img_highres,
        matrix_highres,
        (ref_img_highres.shape[1], ref_img_highres.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned_highres, ["整图ECC对齐"], float(ecc_score)


def warp_mask(mask, matrix, output_shape):
    out_h, out_w = output_shape[:2]
    return cv2.warpAffine(
        mask,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def is_transform_reasonable(matrix, canvas_shape):
    if matrix is None or not np.isfinite(matrix).all():
        return False

    scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    if not (0.2 <= scale_x <= 6.0 and 0.2 <= scale_y <= 6.0):
        return False

    height, width = canvas_shape[:2]
    tx, ty = matrix[0, 2], matrix[1, 2]
    return abs(tx) <= width * 1.7 and abs(ty) <= height * 1.7


def refine_transform_with_sift(org_img_small, ref_img_small, org_mask_small, ref_mask_small, init_matrix):
    h_ref, w_ref = ref_img_small.shape[:2]
    warped_org = cv2.warpAffine(
        org_img_small,
        init_matrix,
        (w_ref, h_ref),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    warped_mask = cv2.warpAffine(
        org_mask_small,
        init_matrix,
        (w_ref, h_ref),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    ref_gray = build_sift_image(ref_img_small)
    warped_gray = build_sift_image(warped_org)
    valid_org_mask = cv2.bitwise_and(ref_mask_small, warped_mask)
    if cv2.countNonZero(valid_org_mask) < 500:
        valid_org_mask = cv2.bitwise_or(ref_mask_small, warped_mask)

    sift = cv2.SIFT_create(
        nfeatures=6000,
        contrastThreshold=0.008,
        edgeThreshold=10,
        sigma=1.2,
    )
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, ref_mask_small)
    kp_org, des_org = sift.detectAndCompute(warped_gray, valid_org_mask)

    if des_ref is None or des_org is None or len(kp_ref) < 8 or len(kp_org) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des_ref, des_org, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.74 * second.distance:
            good_matches.append(first)

    if len(good_matches) < 8:
        return None

    src_pts = np.float32([kp_org[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    refine_matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=3500,
        confidence=0.995,
    )

    if refine_matrix is None or inliers is None or int(inliers.sum()) < 6:
        return None

    combined_matrix = compose_affine(refine_matrix.astype(np.float32), init_matrix)
    if not is_transform_reasonable(combined_matrix, ref_img_small.shape):
        return None

    return combined_matrix


def refine_transform_with_ecc(org_img_small, ref_img_small, org_mask_small, ref_mask_small, init_matrix):
    h_ref, w_ref = ref_img_small.shape[:2]
    warped_org = cv2.warpAffine(
        org_img_small,
        init_matrix,
        (w_ref, h_ref),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    warped_mask = cv2.warpAffine(
        org_mask_small,
        init_matrix,
        (w_ref, h_ref),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if cv2.countNonZero(warped_mask) < 300 or cv2.countNonZero(ref_mask_small) < 300:
        return None

    ref_map = build_ecc_map(ref_img_small, ref_mask_small)
    warped_map = build_ecc_map(warped_org, warped_mask)
    ecc_mask = cv2.dilate(ref_mask_small, np.ones((9, 9), np.uint8), iterations=1)

    refine_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)

    try:
        score, refine_matrix = cv2.findTransformECC(
            ref_map,
            warped_map,
            refine_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=ecc_mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None

    if score < 0.45:
        return None

    combined_matrix = compose_affine(refine_matrix.astype(np.float32), init_matrix)
    if not is_transform_reasonable(combined_matrix, ref_img_small.shape):
        return None

    return combined_matrix


def lock_aspect_ratio(matrix):
    scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    avg_scale = (scale_x + scale_y) / 2.0
    rotation_angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    matrix[0, 0] = avg_scale * np.cos(rotation_angle)
    matrix[0, 1] = -avg_scale * np.sin(rotation_angle)
    matrix[1, 0] = avg_scale * np.sin(rotation_angle)
    matrix[1, 1] = avg_scale * np.cos(rotation_angle)
    return matrix


def warp_with_antialias(org_img_highres, matrix_final, output_size):
    scale_x = np.sqrt(matrix_final[0, 0] ** 2 + matrix_final[0, 1] ** 2)
    scale_y = np.sqrt(matrix_final[1, 0] ** 2 + matrix_final[1, 1] ** 2)
    avg_scale = (scale_x + scale_y) / 2.0
    input_img = org_img_highres
    interpolation = cv2.INTER_CUBIC if avg_scale >= 1.0 else cv2.INTER_LINEAR

    if avg_scale < 0.95:
        sigma = min(1.2, max(0.0, (1.0 / max(avg_scale, 1e-4) - 1.0) * 0.75))
        if sigma > 0.05:
            input_img = cv2.GaussianBlur(input_img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return cv2.warpAffine(
        input_img,
        matrix_final,
        output_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def crop_image_with_box(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2].copy()


def measure_blank_edge_thickness(mask):
    def count_rows(arr):
        thickness = 0
        for row in arr:
            if np.mean(row > 0) >= 0.995:
                break
            thickness += 1
        return thickness

    def count_cols(arr):
        thickness = 0
        for col in arr.transpose(1, 0):
            if np.mean(col > 0) >= 0.995:
                break
            thickness += 1
        return thickness

    return (
        count_rows(mask),
        count_rows(mask[::-1]),
        count_cols(mask),
        count_cols(mask[:, ::-1]),
    )


def measure_white_edge_thickness(img, threshold=246, min_white_ratio=0.985):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def count_rows(arr):
        thickness = 0
        for row in arr:
            if np.mean(np.all(row >= threshold, axis=1)) < min_white_ratio:
                break
            thickness += 1
        return thickness

    def count_cols(arr):
        thickness = 0
        for col in arr.transpose(1, 0, 2):
            if np.mean(np.all(col >= threshold, axis=1)) < min_white_ratio:
                break
            thickness += 1
        return thickness

    return (
        count_rows(rgb),
        count_rows(rgb[::-1]),
        count_cols(rgb),
        count_cols(rgb[:, ::-1]),
    )


def trim_white_edges_and_resize(img, target_size, max_iterations=2):
    result = img
    total_trim = (0, 0, 0, 0)

    for _ in range(max_iterations):
        top, bottom, left, right = measure_white_edge_thickness(result)
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            break

        height, width = result.shape[:2]
        max_trim_h = int(height * 0.12)
        max_trim_w = int(width * 0.12)
        top = min(top, max_trim_h)
        bottom = min(bottom, max_trim_h)
        left = min(left, max_trim_w)
        right = min(right, max_trim_w)

        if top + bottom >= height - 20 or left + right >= width - 20:
            break

        y1 = top
        y2 = height - bottom
        x1 = left
        x2 = width - right
        trimmed = result[y1:y2, x1:x2]
        if trimmed.size == 0:
            break

        total_trim = (
            total_trim[0] + top,
            total_trim[1] + bottom,
            total_trim[2] + left,
            total_trim[3] + right,
        )
        result = cv2.resize(trimmed, target_size, interpolation=cv2.INTER_CUBIC)

    return result, total_trim


def clamp_crop_box(center_x, center_y, crop_w, crop_h, img_w, img_h):
    crop_w = int(round(min(max(crop_w, 32), img_w)))
    crop_h = int(round(min(max(crop_h, 32), img_h)))
    x1 = int(round(center_x - crop_w / 2))
    y1 = int(round(center_y - crop_h / 2))
    x1 = max(0, min(x1, img_w - crop_w))
    y1 = max(0, min(y1, img_h - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def needs_focus_crop(org_mask, ref_mask, org_shape, ref_shape):
    _, _, org_w, org_h = get_mask_bbox(org_mask)
    _, _, ref_w, ref_h = get_mask_bbox(ref_mask)
    org_height_ratio = org_h / max(org_shape[0], 1)
    ref_height_ratio = ref_h / max(ref_shape[0], 1)
    org_width_ratio = org_w / max(org_shape[1], 1)
    ref_width_ratio = ref_w / max(ref_shape[1], 1)
    org_area_ratio = mask_area_ratio(org_mask)
    ref_area_ratio = mask_area_ratio(ref_mask)

    return (
        org_height_ratio < ref_height_ratio * 0.88
        or org_width_ratio < ref_width_ratio * 0.82
        or org_area_ratio < ref_area_ratio * 0.72
    )


def quick_candidate_score(org_mask, ref_mask):
    small_org_mask, _ = resize_with_max_side(org_mask, max_side=1000, interpolation=cv2.INTER_NEAREST)
    small_ref_mask, _ = resize_with_max_side(ref_mask, max_side=1000, interpolation=cv2.INTER_NEAREST)
    matrix_small = estimate_initial_transform_from_masks(small_org_mask, small_ref_mask)
    warped_org_mask = warp_mask(small_org_mask, matrix_small, small_ref_mask.shape)
    return score_mask_alignment(warped_org_mask, small_ref_mask)


def generate_focus_candidates(org_img_highres, org_mask_highres, ref_img_highres, ref_mask_highres, limit=3):
    img_h, img_w = org_img_highres.shape[:2]
    _, mask_y, mask_w, mask_h = get_mask_bbox(org_mask_highres)
    mask_cx, _ = get_mask_centroid(org_mask_highres)
    ref_ratio = ref_img_highres.shape[1] / max(ref_img_highres.shape[0], 1)

    candidate_records = []
    seen_boxes = set()
    for center_ratio in (0.36, 0.44, 0.52, 0.60):
        for height_ratio in (0.34, 0.44, 0.56, 0.70):
            for x_shift in (0.0, -0.05, 0.05):
                crop_h = max(mask_h * height_ratio, img_h * 0.18)
                crop_w = max(crop_h * ref_ratio, mask_w * 0.45)
                center_x = mask_cx + x_shift * mask_w
                center_y = mask_y + center_ratio * mask_h
                box = clamp_crop_box(center_x, center_y, crop_w, crop_h, img_w, img_h)
                if box in seen_boxes:
                    continue
                seen_boxes.add(box)

                candidate_img = crop_image_with_box(org_img_highres, box)
                candidate_mask = crop_image_with_box(org_mask_highres, box)
                if cv2.countNonZero(candidate_mask) == 0:
                    continue

                candidate_records.append(
                    {
                        "box": box,
                        "img": candidate_img,
                        "mask": candidate_mask,
                        "label": f"focus-crop y{center_ratio:.2f} h{height_ratio:.2f}",
                        "quick_score": quick_candidate_score(candidate_mask, ref_mask_highres),
                    }
                )

    grouped_candidates = {}
    for item in candidate_records:
        grouped_candidates.setdefault(item["label"], []).append(item)

    diversified_records = []
    for group_items in grouped_candidates.values():
        group_items.sort(key=lambda item: item["quick_score"], reverse=True)
        diversified_records.extend(group_items[:2])

    diversified_records.sort(key=lambda item: item["quick_score"], reverse=True)
    return diversified_records[:limit]


def score_alignment_quality(result_img, ref_img, ref_mask_highres=None):
    if ref_mask_highres is None:
        ref_mask_highres = extract_foreground_mask(ref_img, allow_grabcut=True)
    result_mask = extract_foreground_mask(result_img, allow_grabcut=False)
    return score_mask_alignment(result_mask, ref_mask_highres)


def score_gray_structure_similarity(candidate_img, ref_crop_img):
    candidate_resized = cv2.resize(
        candidate_img,
        (ref_crop_img.shape[1], ref_crop_img.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    cand_gray_u8 = build_sift_image(candidate_resized)
    ref_gray_u8 = build_sift_image(ref_crop_img)
    cand_gray = cand_gray_u8.astype(np.float32)
    ref_gray = ref_gray_u8.astype(np.float32)
    cand_gray = (cand_gray - cand_gray.mean()) / (cand_gray.std() + 1e-6)
    ref_gray = (ref_gray - ref_gray.mean()) / (ref_gray.std() + 1e-6)
    gray_corr = float(np.clip(np.mean(cand_gray * ref_gray), -1.0, 1.0))
    gray_corr = (gray_corr + 1.0) / 2.0

    cand_edges = cv2.Canny(cand_gray_u8, 40, 120)
    ref_edges = cv2.Canny(ref_gray_u8, 40, 120)
    cand_edges = cand_edges.astype(np.float32)
    ref_edges = ref_edges.astype(np.float32)
    cand_edges = (cand_edges - cand_edges.mean()) / (cand_edges.std() + 1e-6)
    ref_edges = (ref_edges - ref_edges.mean()) / (ref_edges.std() + 1e-6)
    edge_corr = float(np.clip(np.mean(cand_edges * ref_edges), -1.0, 1.0))
    edge_corr = (edge_corr + 1.0) / 2.0
    return 0.80 * gray_corr + 0.20 * edge_corr


def score_same_image_crop_similarity(candidate_img, ref_crop_img):
    candidate_resized = cv2.resize(
        candidate_img,
        (ref_crop_img.shape[1], ref_crop_img.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    gray_score = score_gray_structure_similarity(candidate_img, ref_crop_img)
    cand_mask = extract_foreground_mask(candidate_resized, allow_grabcut=True)
    ref_mask = extract_foreground_mask(ref_crop_img, allow_grabcut=True)
    mask_score = score_mask_alignment(cand_mask, ref_mask)
    return 0.60 * gray_score + 0.40 * mask_score


def build_light_background_layout_mask(img):
    diff_from_white = 255 - img.astype(np.int16)
    max_diff = diff_from_white.max(axis=2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.where(
        (max_diff > 16) | (gray < 240) | (hsv[:, :, 1] > 20),
        255,
        0,
    ).astype(np.uint8)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)


def white_border_ratio(img, threshold=245):
    top = img[0, :, :]
    bottom = img[-1, :, :]
    left = img[:, 0, :]
    right = img[:, -1, :]
    border = np.concatenate([top, bottom, left, right], axis=0)
    return float(np.mean(np.all(border >= threshold, axis=1)))


def find_layout_components(img, min_area_ratio=0.01, max_components=6):
    mask = build_light_background_layout_mask(img)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape[:2]
    min_area = height * width * min_area_ratio
    components = []

    for label in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[label]
        if area < min_area or comp_w < 24 or comp_h < 24:
            continue
        components.append(
            {
                "box": (float(x), float(y), float(comp_w), float(comp_h)),
                "area": float(area),
                "center": (float(centroids[label][0]), float(centroids[label][1])),
                "aspect": float(comp_w) / max(float(comp_h), 1.0),
            }
        )

    components.sort(key=lambda component: component["area"], reverse=True)
    return components[:max_components], mask


def is_light_background_layout(img, components=None, mask=None):
    if components is None or mask is None:
        components, mask = find_layout_components(img)

    area_ratio = mask_area_ratio(mask)
    return (
        len(components) >= 2
        and 0.03 <= area_ratio <= 0.65
        and white_border_ratio(img) >= 0.78
    )


def estimate_layout_similarity_transform(src_components, dst_components, dst_shape):
    match_count = min(len(src_components), len(dst_components), 5)
    if match_count < 2:
        return None, 0.0, ""

    src_candidates = src_components[:match_count]
    dst_candidates = dst_components[: max(match_count, min(len(dst_components), 6))]
    dst_diag = max(float(np.hypot(dst_shape[1], dst_shape[0])), 1.0)
    best = None

    for perm in permutations(range(len(dst_candidates)), match_count):
        paired_dst = [dst_candidates[index] for index in perm]
        scales = []
        shape_costs = []

        for src_component, dst_component in zip(src_candidates, paired_dst):
            _, _, src_w, src_h = src_component["box"]
            _, _, dst_w, dst_h = dst_component["box"]
            if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
                continue
            scales.extend([dst_w / src_w, dst_h / src_h])
            shape_costs.append(
                min(
                    abs(np.log(src_component["aspect"] / max(dst_component["aspect"], 1e-6))),
                    2.0,
                )
            )

        if len(scales) < match_count:
            continue

        scale = float(np.median(scales))
        if scale <= 0.05 or scale >= 5.0:
            continue

        tx = float(
            np.median(
                [
                    dst_component["center"][0] - scale * src_component["center"][0]
                    for src_component, dst_component in zip(src_candidates, paired_dst)
                ]
            )
        )
        ty = float(
            np.median(
                [
                    dst_component["center"][1] - scale * src_component["center"][1]
                    for src_component, dst_component in zip(src_candidates, paired_dst)
                ]
            )
        )

        center_costs = []
        size_costs = []
        for src_component, dst_component in zip(src_candidates, paired_dst):
            src_x, src_y, src_w, src_h = src_component["box"]
            dst_x, dst_y, dst_w, dst_h = dst_component["box"]
            pred_cx = scale * src_component["center"][0] + tx
            pred_cy = scale * src_component["center"][1] + ty
            center_costs.append(
                float(np.hypot(pred_cx - dst_component["center"][0], pred_cy - dst_component["center"][1]) / dst_diag)
            )
            size_costs.append(
                0.5
                * (
                    abs(np.log(max(scale * src_w, 1.0) / max(dst_w, 1.0)))
                    + abs(np.log(max(scale * src_h, 1.0) / max(dst_h, 1.0)))
                )
            )

        cost = (
            float(np.mean(center_costs))
            + float(np.mean(size_costs))
            + 0.20 * float(np.mean(shape_costs or [0.0]))
        )
        if best is None or cost < best[0]:
            best = (cost, scale, tx, ty, perm)

    if best is None:
        return None, 0.0, ""

    cost, scale, tx, ty, perm = best
    score = float(np.clip(1.0 - cost, 0.0, 1.0))
    matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    label = f"layout-similarity s{scale:.3f} ({tx:.0f},{ty:.0f})"
    return matrix, score, label


def process_light_background_layout(org_img_highres, ref_crop_img, ref_full_img):
    org_components, org_mask = find_layout_components(org_img_highres)
    ref_full_components, ref_full_mask = find_layout_components(ref_full_img)
    ref_crop_components, ref_crop_mask = find_layout_components(ref_crop_img)

    if not (
        is_light_background_layout(org_img_highres, org_components, org_mask)
        and is_light_background_layout(ref_full_img, ref_full_components, ref_full_mask)
        and is_light_background_layout(ref_crop_img, ref_crop_components, ref_crop_mask)
    ):
        return None, "", {}

    ref_to_crop_matrix, ref_to_crop_score, ref_to_crop_label = estimate_layout_similarity_transform(
        ref_full_components,
        ref_crop_components,
        ref_crop_img.shape,
    )
    if ref_to_crop_matrix is None or ref_to_crop_score < 0.82:
        return None, "", {}

    org_to_ref_matrix, org_to_ref_score, org_to_ref_label = estimate_layout_similarity_transform(
        org_components,
        ref_full_components,
        ref_full_img.shape,
    )
    if org_to_ref_matrix is not None and org_to_ref_score >= 0.78:
        matrix_final = compose_affine(ref_to_crop_matrix, org_to_ref_matrix)
        steps = [org_to_ref_label, ref_to_crop_label]
        full_align_score = org_to_ref_score
    else:
        matrix_final = ref_to_crop_matrix
        steps = [ref_to_crop_label]
        full_align_score = 1.0

    result = warp_with_antialias(
        org_img_highres,
        matrix_final,
        (ref_crop_img.shape[1], ref_crop_img.shape[0]),
    )
    result_mask = build_light_background_layout_mask(result)
    layout_score = score_mask_alignment(result_mask, ref_crop_mask)
    if layout_score < 0.72:
        return None, "", {}

    details = {
        "score": layout_score,
        "label": ref_to_crop_label,
        "steps": steps,
        "crop_score": ref_to_crop_score,
        "full_align_score": full_align_score,
        "structure_score": layout_score,
        "refine_blank_edges": (),
        "trimmed_white_edges": (0, 0, 0, 0),
    }
    message = f"Success (light background layout: {' + '.join(steps)}, score {layout_score:.3f})"
    return result, message, details


def fit_full_frame_to_reference_canvas(img, ref_full_img, ref_crop_img):
    full_h, full_w = ref_full_img.shape[:2]
    target_h, target_w = ref_crop_img.shape[:2]
    scale = min(target_w / max(full_w, 1), target_h / max(full_h, 1))
    if scale <= 0 or scale > 1.0001:
        return None, 0.0, ""

    new_w = int(round(full_w * scale))
    new_h = int(round(full_h * scale))
    if new_w <= 0 or new_h <= 0 or new_w > target_w or new_h > target_h:
        return None, 0.0, ""

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    def fit_to_canvas(source_img):
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(source_img, (new_w, new_h), interpolation=interpolation)
        canvas = np.full((target_h, target_w, source_img.shape[2]), 255, dtype=source_img.dtype)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas

    fitted_ref = fit_to_canvas(ref_full_img)
    template_score = score_alignment_quality(fitted_ref, ref_crop_img)
    if template_score < 0.92:
        return None, template_score, ""

    fitted_img = fit_to_canvas(img)
    label = f"full-frame resize-pad {scale:.3f} ({pad_x},{pad_y})"
    return fitted_img, template_score, label


def has_extra_white_edges(result_img, ref_img, tolerance=12):
    result_edges = measure_white_edge_thickness(result_img)
    ref_edges = measure_white_edge_thickness(ref_img)
    return any(result > ref + tolerance for result, ref in zip(result_edges, ref_edges))


def locate_crop_box_via_same_image_features(ref_full_img, ref_crop_img):
    full_small, scale_down_full = resize_with_max_side(ref_full_img, max_side=1400, interpolation=cv2.INTER_AREA)
    crop_small, _ = resize_with_max_side(ref_crop_img, max_side=800, interpolation=cv2.INTER_AREA)
    full_gray = build_sift_image(full_small)
    crop_gray = build_sift_image(crop_small)

    sift = cv2.SIFT_create(nfeatures=4000)
    kp_crop, des_crop = sift.detectAndCompute(crop_gray, None)
    kp_full, des_full = sift.detectAndCompute(full_gray, None)
    if des_crop is None or des_full is None or len(kp_crop) < 8 or len(kp_full) < 8:
        return None, 0.0, "same-image feature matching failed"

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des_crop, des_full, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    if len(good_matches) < 10:
        return None, 0.0, "same-image feature matching failed"

    src_pts = np.float32([kp_crop[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_full[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if homography is None or inliers is None or int(inliers.sum()) < 8:
        return None, 0.0, "same-image homography failed"

    crop_h, crop_w = crop_small.shape[:2]
    crop_corners = np.float32(
        [[[0.0, 0.0]], [[crop_w, 0.0]], [[crop_w, crop_h]], [[0.0, crop_h]]]
    )
    projected = cv2.perspectiveTransform(crop_corners, homography).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None, 0.0, "same-image homography failed"

    x_min, y_min = projected.min(axis=0)
    x_max, y_max = projected.max(axis=0)
    if x_max <= x_min or y_max <= y_min:
        return None, 0.0, "same-image homography failed"

    box = clamp_crop_box(
        center_x=(x_min + x_max) * 0.5 * scale_down_full,
        center_y=(y_min + y_max) * 0.5 * scale_down_full,
        crop_w=(x_max - x_min) * scale_down_full,
        crop_h=(y_max - y_min) * scale_down_full,
        img_w=ref_full_img.shape[1],
        img_h=ref_full_img.shape[0],
    )
    candidate = crop_image_with_box(ref_full_img, box)
    score = score_gray_structure_similarity(candidate, ref_crop_img)
    inlier_ratio = int(inliers.sum()) / max(len(good_matches), 1)
    score = 0.70 * score + 0.30 * inlier_ratio
    return box, score, f"same-image feature crop ({int(inliers.sum())}/{len(good_matches)})"


def find_crop_box_from_uncut_template(ref_full_img, ref_crop_img):
    direct_box, direct_score, direct_label = locate_crop_box_via_same_image_features(ref_full_img, ref_crop_img)
    if direct_box is not None and direct_score >= 0.55:
        return direct_box, direct_score, direct_label

    full_small, scale_down = resize_with_max_side(ref_full_img, max_side=1200, interpolation=cv2.INTER_AREA)
    crop_small = cv2.resize(ref_crop_img, (480, 600), interpolation=cv2.INTER_AREA)
    full_mask_small = extract_foreground_mask(full_small, allow_grabcut=True)
    img_h, img_w = full_small.shape[:2]
    _, mask_y, mask_w, mask_h = get_mask_bbox(full_mask_small)
    mask_cx, _ = get_mask_centroid(full_mask_small)
    ref_ratio = crop_small.shape[1] / max(crop_small.shape[0], 1)
    crop_small_gray = build_sift_image(crop_small).astype(np.float32)
    crop_small_gray = (crop_small_gray - crop_small_gray.mean()) / (crop_small_gray.std() + 1e-6)

    best_score = -1.0
    best_box_small = None
    best_label = ""

    for center_ratio in (0.30, 0.36, 0.44, 0.52, 0.60):
        for height_ratio in (0.34, 0.44, 0.56, 0.70):
            for x_shift in (0.0, -0.05, 0.05):
                crop_h = max(mask_h * height_ratio, img_h * 0.18)
                crop_w = max(crop_h * ref_ratio, mask_w * 0.45)
                box_small = clamp_crop_box(
                    mask_cx + x_shift * mask_w,
                    mask_y + center_ratio * mask_h,
                    crop_w,
                    crop_h,
                    img_w,
                    img_h,
                )
                candidate_small = crop_image_with_box(full_small, box_small)
                candidate_small = cv2.resize(
                    candidate_small,
                    (crop_small.shape[1], crop_small.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                candidate_gray = build_sift_image(candidate_small).astype(np.float32)
                candidate_gray = (candidate_gray - candidate_gray.mean()) / (candidate_gray.std() + 1e-6)
                score = float(np.clip(np.mean(candidate_gray * crop_small_gray), -1.0, 1.0))
                if score > best_score:
                    best_score = score
                    best_box_small = box_small
                    best_label = f"crop-box y{center_ratio:.2f} h{height_ratio:.2f}"

    if best_box_small is None:
        return None, 0.0, "crop-box failed"

    scale = ref_full_img.shape[1] / max(full_small.shape[1], 1)
    best_box = tuple(int(round(value * scale)) for value in best_box_small)
    return best_box, best_score, best_label


def refine_scene_crop_with_masks(cropped_result, ref_crop_img):
    org_mask = extract_foreground_mask(cropped_result, allow_grabcut=False)
    ref_mask = extract_foreground_mask(ref_crop_img, allow_grabcut=False)
    if cv2.countNonZero(org_mask) == 0 or cv2.countNonZero(ref_mask) == 0:
        return None, [], {}

    crop_small, org_mask_small, scale_down_org = resize_image_and_mask(
        cropped_result,
        org_mask,
        max_size=1000,
    )
    ref_small, ref_mask_small, scale_down_ref = resize_image_and_mask(
        ref_crop_img,
        ref_mask,
        max_size=1000,
    )
    matrix_small = estimate_initial_transform_from_masks(org_mask_small, ref_mask_small)
    matrix_highres = scale_affine_from_small_to_large(matrix_small, scale_down_org, scale_down_ref)
    matrix_highres = lock_aspect_ratio(matrix_highres)
    if not is_transform_reasonable(matrix_highres, ref_crop_img.shape):
        return None, [], {}

    coverage_mask = np.full(cropped_result.shape[:2], 255, dtype=np.uint8)
    warped_coverage_mask = cv2.warpAffine(
        coverage_mask,
        matrix_highres,
        (ref_crop_img.shape[1], ref_crop_img.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    blank_edges = measure_blank_edge_thickness(warped_coverage_mask)
    if any(blank_edges):
        return None, [], {"blank_edges": blank_edges}

    result = warp_with_antialias(cropped_result, matrix_highres, (ref_crop_img.shape[1], ref_crop_img.shape[0]))
    return result, ["前景框微调"], {"blank_edges": blank_edges}


def process_with_uncut_template(org_img_highres, ref_crop_img, ref_full_img, return_details=False):
    if org_img_highres is None or ref_crop_img is None or ref_full_img is None:
        message = "场景图模式缺少必要图片"
        if return_details:
            return None, message, {}
        return None, message

    layout_result, layout_message, layout_details = process_light_background_layout(
        org_img_highres,
        ref_crop_img,
        ref_full_img,
    )
    if layout_result is not None:
        if return_details:
            return layout_result, layout_message, layout_details
        return layout_result, layout_message

    canvas_result, canvas_template_score, canvas_label = fit_full_frame_to_reference_canvas(
        org_img_highres,
        ref_full_img,
        ref_crop_img,
    )
    if canvas_result is not None:
        canvas_score = score_alignment_quality(canvas_result, ref_crop_img)
        if canvas_score >= 0.94:
            steps = [canvas_label]
            message = f"Success (scene mode: {' + '.join(steps)}, score {canvas_score:.3f})"
            details = {
                "score": canvas_score,
                "label": canvas_label,
                "steps": steps,
                "crop_score": canvas_template_score,
                "full_align_score": 1.0,
                "structure_score": score_gray_structure_similarity(canvas_result, ref_crop_img),
                "refine_blank_edges": (),
                "trimmed_white_edges": (0, 0, 0, 0),
            }
            if return_details:
                return canvas_result, message, details
            return canvas_result, message

    aligned_full, full_steps, full_align_score = align_scene_full_frame_fast(org_img_highres, ref_full_img)
    if aligned_full is None:
        aligned_full, full_steps = align_single_candidate(org_img_highres, ref_full_img)
        full_align_score = 0.60
    if aligned_full is None:
        message = "无法把原图对齐到未截头模板"
        if return_details:
            return None, message, {}
        return None, message

    crop_box, crop_score, crop_label = find_crop_box_from_uncut_template(ref_full_img, ref_crop_img)
    if crop_box is None:
        message = "无法从未截头模板反推出截图框"
        if return_details:
            return None, message, {}
        return None, message

    cropped_result = crop_image_with_box(aligned_full, crop_box)
    cropped_result = cv2.resize(
        cropped_result,
        (ref_crop_img.shape[1], ref_crop_img.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    best_result = cropped_result
    best_steps = [f"整图对齐({' + '.join(full_steps)})", crop_label]
    best_score = 0.55 * crop_score + 0.45 * full_align_score
    best_structure_score = score_gray_structure_similarity(cropped_result, ref_crop_img)

    refined_result, refined_steps, refined_meta = refine_scene_crop_with_masks(cropped_result, ref_crop_img)
    if refined_result is not None:
        refined_structure_score = score_gray_structure_similarity(refined_result, ref_crop_img)
        if refined_structure_score > best_structure_score + 0.005:
            best_result = refined_result
            best_steps = [f"整图对齐({' + '.join(full_steps)})", crop_label] + refined_steps
            best_structure_score = refined_structure_score

    trimmed_white_edges = (0, 0, 0, 0)
    if has_extra_white_edges(best_result, ref_crop_img):
        current_quality_score = score_alignment_quality(best_result, ref_crop_img)
        trim_candidate, candidate_trimmed_white_edges = trim_white_edges_and_resize(
            best_result,
            (ref_crop_img.shape[1], ref_crop_img.shape[0]),
        )
        trim_quality_score = score_alignment_quality(trim_candidate, ref_crop_img)
        if trim_quality_score > current_quality_score + 0.005:
            best_result = trim_candidate
            trimmed_white_edges = candidate_trimmed_white_edges
    if any(trimmed_white_edges):
        best_steps = best_steps + ["白边兜底裁切"]
        best_structure_score = score_gray_structure_similarity(best_result, ref_crop_img)

    best_score = score_alignment_quality(best_result, ref_crop_img)

    message = f"成功（场景图模式：{' + '.join(best_steps)}，评分 {best_score:.3f}）"
    details = {
        "score": best_score,
        "label": crop_label,
        "steps": best_steps,
        "crop_score": crop_score,
        "full_align_score": full_align_score,
        "structure_score": best_structure_score,
        "refine_blank_edges": refined_meta.get("blank_edges", ()) if isinstance(refined_meta, dict) else (),
        "trimmed_white_edges": trimmed_white_edges,
    }
    if return_details:
        return best_result, message, details
    return best_result, message


def align_single_candidate(org_img_highres, ref_img_highres, org_mask_highres=None, ref_mask_highres=None):
    if org_img_highres is None or ref_img_highres is None:
        return None, []

    if org_mask_highres is None:
        org_mask_highres = extract_foreground_mask(org_img_highres, allow_grabcut=True)
    if ref_mask_highres is None:
        ref_mask_highres = extract_foreground_mask(ref_img_highres, allow_grabcut=True)

    h_ref, w_ref = ref_img_highres.shape[:2]
    org_img_small, org_mask_small, scale_down_org = resize_image_and_mask(
        org_img_highres,
        org_mask_highres,
        max_size=1200,
    )
    ref_img_small, ref_mask_small, scale_down_ref = resize_image_and_mask(
        ref_img_highres,
        ref_mask_highres,
        max_size=1200,
    )

    initial_matrix = estimate_initial_transform_from_masks(org_mask_small, ref_mask_small)
    best_matrix_small = initial_matrix
    best_steps = ["前景包围盒预对齐"]
    best_mask_score = score_mask_alignment(
        warp_mask(org_mask_small, initial_matrix, ref_mask_small.shape),
        ref_mask_small,
    )

    sift_matrix = refine_transform_with_sift(
        org_img_small,
        ref_img_small,
        org_mask_small,
        ref_mask_small,
        initial_matrix,
    )
    sift_score = None
    if sift_matrix is not None:
        sift_score = score_mask_alignment(
            warp_mask(org_mask_small, sift_matrix, ref_mask_small.shape),
            ref_mask_small,
        )
        if sift_score > best_mask_score:
            best_matrix_small = sift_matrix
            best_steps = ["前景包围盒预对齐", "SIFT微调"]
            best_mask_score = sift_score

    ecc_seeds = [(initial_matrix, ["前景包围盒预对齐"])]
    if sift_matrix is not None and sift_score is not None and sift_score >= best_mask_score - 1e-6:
        ecc_seeds.append((sift_matrix, ["前景包围盒预对齐", "SIFT微调"]))

    for seed_matrix, seed_steps in ecc_seeds:
        ecc_matrix = refine_transform_with_ecc(
            org_img_small,
            ref_img_small,
            org_mask_small,
            ref_mask_small,
            seed_matrix,
        )
        if ecc_matrix is None:
            continue

        ecc_score = score_mask_alignment(
            warp_mask(org_mask_small, ecc_matrix, ref_mask_small.shape),
            ref_mask_small,
        )
        if ecc_score > best_mask_score:
            best_matrix_small = ecc_matrix
            best_steps = seed_steps + ["ECC轮廓微调"]
            best_mask_score = ecc_score

    matrix_highres = scale_affine_from_small_to_large(
        best_matrix_small,
        scale_down_org,
        scale_down_ref,
    )
    matrix_highres = lock_aspect_ratio(matrix_highres)
    if not is_transform_reasonable(matrix_highres, ref_img_highres.shape):
        return None, best_steps

    result_highres = warp_with_antialias(org_img_highres, matrix_highres, (w_ref, h_ref))
    return result_highres, best_steps


def align_and_crop_strict(org_img_highres, ref_img_highres, return_details=False):
    if org_img_highres is None or ref_img_highres is None:
        message = "图片读取失败，请确认文件是否损坏"
        if return_details:
            return None, message, {}
        return None, message

    org_mask_highres = extract_foreground_mask(org_img_highres, allow_grabcut=True)
    ref_mask_highres = extract_foreground_mask(ref_img_highres, allow_grabcut=True)

    best_result = None
    best_steps = []
    best_score = -1.0
    best_label = "full-frame"

    full_result, full_steps = align_single_candidate(
        org_img_highres,
        ref_img_highres,
        org_mask_highres=org_mask_highres,
        ref_mask_highres=ref_mask_highres,
    )
    if full_result is not None:
        full_score = score_alignment_quality(full_result, ref_img_highres, ref_mask_highres=ref_mask_highres)
        best_result = full_result
        best_steps = full_steps
        best_score = full_score

    need_focus = needs_focus_crop(
        org_mask_highres,
        ref_mask_highres,
        org_img_highres.shape,
        ref_img_highres.shape,
    ) or best_score < 0.88

    if need_focus:
        candidate_limit = 18 if best_score < 0.82 else 6
        for candidate in generate_focus_candidates(
            org_img_highres,
            org_mask_highres,
            ref_img_highres,
            ref_mask_highres,
            limit=candidate_limit,
        ):
            candidate_result, candidate_steps = align_single_candidate(
                candidate["img"],
                ref_img_highres,
                org_mask_highres=candidate["mask"],
                ref_mask_highres=ref_mask_highres,
            )
            if candidate_result is None:
                continue

            candidate_score = score_alignment_quality(
                candidate_result,
                ref_img_highres,
                ref_mask_highres=ref_mask_highres,
            )
            if candidate_score > best_score:
                best_result = candidate_result
                best_steps = candidate_steps
                best_score = candidate_score
                best_label = candidate["label"]

            if best_score >= 0.95:
                break

    if best_result is None:
        message = "无法计算稳定对齐结果，请确认模板图和原图主体一致"
        if return_details:
            return None, message, {}
        return None, message

    step_text = " + ".join(best_steps)
    if best_label != "full-frame":
        message = f"成功（{best_label} + {step_text}，评分 {best_score:.3f}）"
    else:
        message = f"成功（{step_text}，评分 {best_score:.3f}）"

    details = {"score": best_score, "label": best_label, "steps": best_steps}
    if return_details:
        return best_result, message, details
    return best_result, message


def process_local_group(group_name, output_format="JPEG"):
    org_dir = TEST_ORG_DIR / group_name
    ref_dir = TEST_REF_DIR / group_name
    out_dir = TEST_OUTPUT_DIR / f"{group_name}_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    org_paths = list_image_files(org_dir)
    ref_paths = list_image_files(ref_dir)
    cropped_refs, uncut_refs = split_local_reference_sets(org_paths, ref_paths)
    triples, pairing_mode, uncut_mode = build_reference_triples(org_paths, cropped_refs, uncut_refs)
    results = []

    for triple in triples:
        org_path = triple["original"]
        ref_path = triple["cropped_ref"]
        uncut_path = triple["uncut_ref"]

        org_img = load_path_image(org_path)
        ref_img = load_path_image(ref_path)
        if uncut_path is not None:
            uncut_img = load_path_image(uncut_path)
            res_img, msg, details = process_with_uncut_template(
                org_img,
                ref_img,
                uncut_img,
                return_details=True,
            )
        else:
            res_img, msg, details = align_and_crop_strict(org_img, ref_img, return_details=True)

        if res_img is None:
            results.append(
                {
                    "original": org_path.name,
                    "template": ref_path.name,
                    "uncut_template": uncut_path.name if uncut_path is not None else "",
                    "status": "FAIL",
                    "message": msg,
                    "score": 0.0,
                    "output": "",
                }
            )
            continue

        img_bytes, extension = convert_cv_to_bytes(res_img, output_format=output_format)
        out_path = out_dir / f"{Path(ref_path).stem}_from_{Path(org_path).stem}.{extension}"
        out_path.write_bytes(img_bytes)
        results.append(
            {
                "original": org_path.name,
                "template": ref_path.name,
                "uncut_template": uncut_path.name if uncut_path is not None else "",
                "status": "OK",
                "message": msg,
                "score": details.get("score", 0.0),
                "output": str(out_path),
            }
        )

    return {
        "group": group_name,
        "pairing_mode": pairing_mode,
        "uncut_mode": uncut_mode,
        "output_dir": str(out_dir),
        "results": results,
    }


def render_uploaded_processing_ui(output_format):
    col1, col2, col3 = st.columns(3)
    with col1:
        org_files = st.file_uploader(
            "1. 上传修后原图",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
    with col2:
        ref_files = st.file_uploader(
            "2. 上传拍图模板图片",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
    with col3:
        uncut_ref_files = st.file_uploader(
            "3. 上传拍图模板（未截头，可选）",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="不上传时走默认算法；上传后，场景图会先对齐到未截头模板，再按模板截图框裁切。",
        )

    if not org_files or not ref_files:
        return

    st.divider()
    triples, pairing_mode, uncut_mode = build_reference_triples(org_files, ref_files, uncut_ref_files or [])
    st.caption(f"当前配对策略：{pairing_mode}")
    if uncut_ref_files:
        st.caption(f"未截头模板配对策略：{uncut_mode}")

    if not triples:
        st.warning("当前没有可处理的配对，请检查上传文件。")
        return

    if st.button("启动处理", type="primary", use_container_width=True):
        zip_buffer = BytesIO()
        success_count = 0

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for triple in triples:
                org_file = triple["original"]
                matched_ref = triple["cropped_ref"]
                matched_uncut = triple["uncut_ref"]
                try:
                    img_org = load_raw_image(org_file)
                    img_ref = load_raw_image(matched_ref)
                    img_uncut = load_raw_image(matched_uncut) if matched_uncut is not None else None

                    with st.spinner(f"正在智能对齐 {org_file.name} ..."):
                        if img_uncut is not None:
                            res_img, msg, details = process_with_uncut_template(
                                img_org,
                                img_ref,
                                img_uncut,
                                return_details=True,
                            )
                        else:
                            res_img, msg, details = align_and_crop_strict(img_org, img_ref, return_details=True)

                    if res_img is None:
                        st.error(f"处理失败：{org_file.name} -> {matched_ref.name}，原因：{msg}")
                        continue

                    ref_name = Path(matched_ref.name).stem
                    img_bytes, extension = convert_cv_to_bytes(res_img, output_format=output_format)
                    file_name = f"{ref_name}.{extension}"
                    zip_file.writestr(file_name, img_bytes)

                    with st.expander(f"已处理：{org_file.name} -> {file_name}", expanded=True):
                        preview_col1, preview_col2, _ = st.columns([1.5, 1.5, 7])
                        with preview_col1:
                            st.markdown("**参考模板图**")
                            preview_ref = cv2.resize(img_ref, (0, 0), fx=0.15, fy=0.15)
                            st.image(cv2.cvtColor(preview_ref, cv2.COLOR_BGR2RGB), width=110)
                        with preview_col2:
                            st.markdown("**处理结果**")
                            preview_res = cv2.resize(res_img, (0, 0), fx=0.15, fy=0.15)
                            st.image(cv2.cvtColor(preview_res, cv2.COLOR_BGR2RGB), width=110)
                        if img_uncut is not None:
                            st.caption(f"{msg}，最终评分 {details.get('score', 0.0):.3f}，使用未截头模板 {matched_uncut.name}")
                        else:
                            st.caption(f"{msg}，最终评分 {details.get('score', 0.0):.3f}")

                    success_count += 1
                except Exception as exc:
                    st.error(f"处理 {org_file.name} 时出错：{str(exc)}")

        if success_count > 0:
            st.divider()
            st.download_button(
                label=f"下载处理完成的 {success_count} 张打包文件",
                data=zip_buffer.getvalue(),
                file_name="strict_aligned_images.zip",
                mime="application/zip",
                use_container_width=True,
            )


def main():
    st.set_page_config(page_title="按着拍图模板自动裁图", page_icon="📏", layout="wide")
    st.title("📏 按着拍图模板自动裁图")
    st.caption("这版支持两条处理路径：默认模板裁切，以及可选的“未截头模板”场景图路径。")

    render_uploaded_processing_ui("JPEG")


if __name__ == "__main__":
    main()
