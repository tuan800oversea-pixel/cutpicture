import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import os
import zipfile
from PIL import Image
import mediapipe as mp

# ================= 图片加载 =================
def load_raw_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def convert_cv_to_bytes(cv_img):
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=100, dpi=(300, 300))
    return buf.getvalue()

def find_matching_reference(org_filename, ref_files):
    org_base = os.path.splitext(org_filename)[0]
    sorted_refs = sorted(ref_files, key=lambda x: len(os.path.splitext(x.name)[0]), reverse=True)
    for ref in sorted_refs:
        ref_base = os.path.splitext(ref.name)[0]
        if org_base == ref_base or org_base.endswith(ref_base):
            return ref
    return None

# ================= 核心算法：人体语义关键点 (锁定旋转) =================
def get_body_keypoints(img):
    mp_pose = mp.solutions.pose
    h, w = img.shape[:2]
    max_calc_size = 800
    scale = max(h, w) / max_calc_size if max(h, w) > max_calc_size else 1.0
    img_small = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        
        lm = results.pose_landmarks.landmark
        # 选取最稳定的三个点：左肩(11), 右肩(12), 髋部中心(23,24)
        # 这三个点构成的三角形最能代表躯干主体的缩放和位置
        pts = np.array([
            [lm[11].x * w, lm[11].y * h], # 左肩
            [lm[12].x * w, lm[12].y * h], # 右肩
            [(lm[23].x + lm[24].x)*0.5 * w, (lm[23].y + lm[24].y)*0.5 * h] # 髋部中点
        ], dtype=np.float32)
        return pts

def align_and_crop_strict(org_img, ref_img):
    h_ref, w_ref = ref_img.shape[:2]
    
    pts_org = get_body_keypoints(org_img)
    pts_ref = get_body_keypoints(ref_img)
    
    if pts_org is None or pts_ref is None:
        return None, "未能检测到完整人体结构"

    # --- 核心改进：计算不含旋转的缩放平移矩阵 ---
    # 计算躯干宽度作为缩放参考
    dist_org = np.linalg.norm(pts_org[0] - pts_org[1])
    dist_ref = np.linalg.norm(pts_ref[0] - pts_ref[1])
    
    # 锁定等比例缩放系数
    s = dist_ref / dist_org
    
    # 计算平移量（以躯干中心点为基准）
    center_org = np.mean(pts_org, axis=0)
    center_ref = np.mean(pts_ref, axis=0)
    
    tx = center_ref[0] - s * center_org[0]
    ty = center_ref[1] - s * center_org[1]

    # 构造强制 [0度旋转] 的矩阵
    # [ s  0  tx ]
    # [ 0  s  ty ]
    M = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float32)

    result = cv2.warpAffine(org_img, M, (w_ref, h_ref), 
                            flags=cv2.INTER_LANCZOS4, 
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(255, 255, 255))
    return result, "成功"

# ================= Streamlit UI (保持原版) =================
st.set_page_config(page_title="按着拍图模板自动裁图", page_icon="📏", layout="wide")
st.title("📏 按着拍图模板自动裁图")

col1, col2 = st.columns(2)
with col1:
    org_files = st.file_uploader("1️⃣ 上传修后原图(待截图)", accept_multiple_files=True)
with col2:
    ref_files = st.file_uploader("2️⃣ 上传拍图模板图片(截图后)", accept_multiple_files=True)

if org_files and ref_files:
    st.divider()
    if st.button("🚀 启动 100% 等比无损处理", type="primary", use_container_width=True):
        zip_buffer = BytesIO()
        success_count = 0
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for org_file in org_files:
                matched_ref = find_matching_reference(org_file.name, ref_files)
                if not matched_ref: continue
                
                try:
                    img_org = load_raw_image(org_file)
                    img_ref = load_raw_image(matched_ref)
                    
                    with st.spinner(f"正在智能对齐 {org_file.name}..."):
                        res_img, msg = align_and_crop_strict(img_org, img_ref)

                    if res_img is not None:
                        img_bytes = convert_cv_to_bytes(res_img)
                        zip_file.writestr(f"{os.path.splitext(matched_ref.name)[0]}.jpg", img_bytes)
                        
                        with st.expander(f"✅ 已处理: {org_file.name}", expanded=True):
                            p1, p2, _ = st.columns([1.5, 1.5, 7])
                            p1.image(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB), width=100, caption="参考模板")
                            p2.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), width=100, caption="最终裁切")
                        success_count += 1
                except Exception as e:
                    st.error(f"处理失败: {str(e)}")

        if success_count > 0:
            st.download_button("📥 下载打包文件", zip_buffer.getvalue(), "aligned.zip", use_container_width=True)
