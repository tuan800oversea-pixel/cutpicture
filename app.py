import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import os
import zipfile
from PIL import Image

# 引入 AI 抠图库
from rembg import remove

# ================= 图片加载 =================
def load_raw_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

# ================= 转换为 300 DPI 的 JPG =================
def convert_cv_to_bytes(cv_img):
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=100, dpi=(300, 300))
    return buf.getvalue()

# ================= 文件名匹配逻辑 =================
def find_matching_reference(org_filename, ref_files):
    org_base = os.path.splitext(org_filename)[0]
    sorted_refs = sorted(ref_files, key=lambda x: len(os.path.splitext(x.name)[0]), reverse=True)
    for ref in sorted_refs:
        ref_base = os.path.splitext(ref.name)[0]
        if org_base == ref_base or org_base.endswith(ref_base):
            return ref
    return None

# ================= 核心算法：AI 抠图提取纯轮廓 (无视背景与内部摩尔纹) =================
def get_ai_silhouette_masks(img_small_color):
    """
    使用 rembg 进行真正的 AI 语义分割，无论什么背景都能精准提取前景(人体/衣服)遮罩。
    """
    # 1. AI 生成精确的前景 Alpha 遮罩 (黑白图：前景白，背景黑)
    # only_mask=True 表示我们只要黑白遮罩，不要抠出来的彩图
    ai_mask = remove(img_small_color, only_mask=True)
    
    # 转换为 OpenCV 标准的单通道图以防万一
    if len(ai_mask.shape) == 3:
        ai_mask = cv2.cvtColor(ai_mask, cv2.COLOR_BGR2GRAY)
        
    gray_img = cv2.cvtColor(img_small_color, cv2.COLOR_BGR2GRAY)
    
    # 2. 提取外轮廓
    contours, _ = cv2.findContours(ai_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- 遮罩 A：边缘搜索带 (Edge Band) ---
    # 沿着 AI 抠出来的边缘画一圈，强迫 SIFT 只能在这个边缘地带找特征，避开内部的摩尔纹水波
    edge_mask = np.zeros_like(gray_img)
    cv2.drawContours(edge_mask, contours, -1, 255, thickness=40)
    
    # --- 遮罩 B：平滑形状斑块 (Blob) ---
    # 纯白色的剪影，用于 SIFT 失败时的极端形状对齐
    blob_mask = np.zeros_like(gray_img)
    cv2.drawContours(blob_mask, contours, -1, 255, thickness=cv2.FILLED)
    blob_mask = cv2.GaussianBlur(blob_mask, (41, 41), 0) # 模糊边缘防锯齿干扰
    blob_mask = np.float32(blob_mask) / 255.0
    
    return gray_img, edge_mask, blob_mask

# ================= 核心对齐：外轮廓限制级对齐引擎 =================
def align_and_crop_strict(org_img_highres, ref_img_highres):
    h_org, w_org = org_img_highres.shape[:2]
    h_ref, w_ref = ref_img_highres.shape[:2]

    # 降维计算，提速并过滤高频噪点
    max_calc_size = 800
    scale_down_org = max(h_org, w_org) / max_calc_size if max(h_org, w_org) > max_calc_size else 1.0
    scale_down_ref = max(h_ref, w_ref) / max_calc_size if max(h_ref, w_ref) > max_calc_size else 1.0

    org_img_small = cv2.resize(org_img_highres, (int(w_org / scale_down_org), int(h_org / scale_down_org)), interpolation=cv2.INTER_AREA)
    ref_img_small = cv2.resize(ref_img_highres, (int(w_ref / scale_down_ref), int(h_ref / scale_down_ref)), interpolation=cv2.INTER_AREA)

    # 传入彩图给 AI 获取精准遮罩
    gray_org, edge_mask_org, blob_org = get_ai_silhouette_masks(org_img_small)
    gray_ref, edge_mask_ref, blob_ref = get_ai_silhouette_masks(ref_img_small)

    M_final = None

    # --- 策略 A：轮廓边缘特征匹配 (SIFT) ---
    sift = cv2.SIFT_create(nfeatures=5000)
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, mask=edge_mask_ref)
    kp_org, des_org = sift.detectAndCompute(gray_org, mask=edge_mask_org)

    if des_org is not None and des_ref is not None and len(kp_org) > 5:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_ref, des_org, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) >= 6:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale_down_ref
            dst_pts = np.float32([kp_org[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale_down_org

            M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if M is not None:
                M_final = M

    # --- 策略 B：纯形状盲眼对齐 (ECC 保底) ---
    if M_final is None:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.0005)
        try:
            _, warp_matrix = cv2.findTransformECC(blob_ref, blob_org, warp_matrix, cv2.MOTION_AFFINE, criteria)
            M_final = warp_matrix.copy()
            scale_ratio = scale_down_ref / scale_down_org
            M_final[0, 0] *= scale_ratio
            M_final[0, 1] *= scale_ratio
            M_final[0, 2] *= scale_down_ref
            M_final[1, 0] *= scale_ratio
            M_final[1, 1] *= scale_ratio
            M_final[1, 2] *= scale_down_ref
        except Exception:
            return None, "轮廓匹配失败：两图的模特身形差异过大或特征被过度破坏"

    if M_final is None:
        return None, "无法计算对齐路径，请检查图片"

    # 锁定纵横比
    s_x = np.sqrt(M_final[0, 0]**2 + M_final[0, 1]**2)
    s_y = np.sqrt(M_final[1, 0]**2 + M_final[1, 1]**2)
    avg_scale = (s_x + s_y) / 2.0
    rotation_angle = np.arctan2(M_final[1, 0], M_final[0, 0])
    
    M_final[0, 0] = avg_scale * np.cos(rotation_angle)
    M_final[0, 1] = -avg_scale * np.sin(rotation_angle)
    M_final[1, 0] = avg_scale * np.sin(rotation_angle)
    M_final[1, 1] = avg_scale * np.cos(rotation_angle)

    # 裁切输出
    result_highres = cv2.warpAffine(org_img_highres, M_final, (w_ref, h_ref), 
                                    flags=cv2.INTER_LANCZOS4, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(255, 255, 255))
    
    return result_highres, "成功"


# ================= Streamlit UI (保持原版不动) =================
st.set_page_config(page_title="按着拍图模板自动裁图", page_icon="📏", layout="wide")

st.title("📏 按着拍图模板自动裁图")

col1, col2 = st.columns(2)
with col1:
    org_files = st.file_uploader("1️⃣ 上传修后原图(待截图)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
with col2:
    ref_files = st.file_uploader("2️⃣ 上传拍图模板图片(截图后)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if org_files and ref_files:
    st.divider()
    if st.button("🚀 启动 100% 等比无损处理", type="primary", use_container_width=True):
        zip_buffer = BytesIO()
        success_count = 0
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for org_file in org_files:
                matched_ref = find_matching_reference(org_file.name, ref_files)
                
                if not matched_ref:
                    st.warning(f"跳过: {org_file.name} (未找到匹配模板)")
                    continue
                
                try:
                    img_org = load_raw_image(org_file)
                    matched_ref.seek(0)
                    img_ref = load_raw_image(matched_ref)
                    
                    with st.spinner(f"正在智能抠图与对齐 {org_file.name} (初次运行需加载AI模型，稍等片刻)..."):
                        res_img, msg = align_and_crop_strict(img_org, img_ref)

                    if res_img is not None:
                        ref_name = os.path.splitext(matched_ref.name)[0]
                        file_name = f"{ref_name}.jpg"
                        
                        img_bytes = convert_cv_to_bytes(res_img)
                        zip_file.writestr(file_name, img_bytes)
                        
                        with st.expander(f"✅ 已处理: {org_file.name} ➔ {file_name}", expanded=True):
                            preview_col1, preview_col2, _ = st.columns([1.5, 1.5, 7])
                            
                            with preview_col1:
                                st.markdown("**🖼️ 参考模板图**")
                                preview_ref = cv2.resize(img_ref, (0,0), fx=0.15, fy=0.15)
                                st.image(cv2.cvtColor(preview_ref, cv2.COLOR_BGR2RGB), width=100)
                                
                            with preview_col2:
                                st.markdown("**✨ 截图后的图片**")
                                preview_res = cv2.resize(res_img, (0,0), fx=0.15, fy=0.15)
                                st.image(cv2.cvtColor(preview_res, cv2.COLOR_BGR2RGB), width=100)
                        
                        success_count += 1
                    else:
                        st.error(f"❌ {org_file.name} 失败: {msg}")
                        
                except Exception as e:
                    st.error(f"⚠️ 处理 {org_file.name} 时出错: {str(e)}")

        if success_count > 0:
            st.divider()
            st.download_button(
                label=f"📥 下载处理完成的 {success_count} 张打包文件",
                data=zip_buffer.getvalue(),
                file_name="strict_aligned_images.zip",
                mime="application/zip",
                use_container_width=True
            )
