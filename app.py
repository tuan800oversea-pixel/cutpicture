import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import os
import zipfile
from PIL import Image

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
    # 强制 300 DPI 输出，确保印刷质量
    pil_img.save(buf, format='JPEG', quality=100, dpi=(300, 300))
    return buf.getvalue()

# ================= 文件名匹配逻辑 =================
def find_matching_reference(org_filename, ref_files):
    org_base = os.path.splitext(org_filename)[0]
    
    # 核心优化：按参考文件名的长度降序排序。
    sorted_refs = sorted(ref_files, key=lambda x: len(os.path.splitext(x.name)[0]), reverse=True)
    
    for ref in sorted_refs:
        ref_base = os.path.splitext(ref.name)[0]
        # 判断逻辑：原图名等于参考图名，或者原图名以参考图名结尾
        if org_base == ref_base or org_base.endswith(ref_base):
            return ref
    return None

# ================= 核心算法：重度抗螺纹高精度对齐 =================
def align_and_crop_strict(org_img_highres, ref_img_highres):
    h_org, w_org = org_img_highres.shape[:2]
    h_ref, w_ref = ref_img_highres.shape[:2]

    # 1. 预处理：降维打击
    # 将计算尺寸从 1200 降到 800，强迫系统忽略细密的截图摩尔纹，只关注大轮廓
    max_calc_size = 800
    scale_down_org = max(h_org, w_org) / max_calc_size if max(h_org, w_org) > max_calc_size else 1.0
    scale_down_ref = max(h_ref, w_ref) / max_calc_size if max(h_ref, w_ref) > max_calc_size else 1.0

    org_img_small = cv2.resize(org_img_highres, (int(w_org / scale_down_org), int(h_org / scale_down_org)), interpolation=cv2.INTER_AREA)
    ref_img_small = cv2.resize(ref_img_highres, (int(w_ref / scale_down_ref), int(h_ref / scale_down_ref)), interpolation=cv2.INTER_AREA)

    # 转为灰度图
    gray_org_small = cv2.cvtColor(org_img_small, cv2.COLOR_BGR2GRAY)
    gray_ref_small = cv2.cvtColor(ref_img_small, cv2.COLOR_BGR2GRAY)
    
    # ================= 核心突破：重度摩尔纹消除组合拳 =================
    # 招式一：中值滤波 (Median Blur)。它是结构性噪声（水波纹、网格）的天然克星
    blur_org = cv2.medianBlur(gray_org_small, 5)
    blur_ref = cv2.medianBlur(gray_ref_small, 5)

    # 招式二：双边滤波 (Bilateral Filter)。给面料做极强力的“磨皮”，但死死保住衣服边缘和系带的锐利度
    blur_org = cv2.bilateralFilter(blur_org, d=9, sigmaColor=75, sigmaSpace=75)
    blur_ref = cv2.bilateralFilter(blur_ref, d=9, sigmaColor=75, sigmaSpace=75)

    # 招式三：CLAHE 增强。把磨皮后仅存的真实结构边缘（比如阴影轮廓）提亮
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # 稍微调低对比度，防止把噪点重新放大
    gray_org_enhanced = clahe.apply(blur_org)
    gray_ref_enhanced = clahe.apply(blur_ref)
    # ====================================================================

    # 2. SIFT 特征提取
    sift = cv2.SIFT_create(nfeatures=10000)
    kp_ref, des_ref = sift.detectAndCompute(gray_ref_enhanced, None)
    kp_org, des_org = sift.detectAndCompute(gray_org_enhanced, None)

    M_final = None

    # 3. 尝试 SIFT 匹配
    if des_org is not None and des_ref is not None and len(kp_org) >= 10:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_ref, des_org, k=2)
        
        # 因为经过了重度滤波，特征点的独特性下降，这里放宽匹配通过率至 0.8
        good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

        if len(good_matches) >= 10:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale_down_ref
            dst_pts = np.float32([kp_org[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale_down_org

            M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if M is not None:
                M_final = M

    # 4. 【兜底方案】ECC 轮廓对齐
    if M_final is None:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        try:
            _, warp_matrix = cv2.findTransformECC(gray_ref_enhanced, gray_org_enhanced, warp_matrix, cv2.MOTION_AFFINE, criteria)
            M_final = warp_matrix.copy()
            scale_ratio = scale_down_ref / scale_down_org
            M_final[0, 0] *= scale_ratio
            M_final[0, 1] *= scale_ratio
            M_final[0, 2] *= scale_down_ref
            M_final[1, 0] *= scale_ratio
            M_final[1, 1] *= scale_ratio
            M_final[1, 2] *= scale_down_ref
        except Exception:
            return None, "匹配点极少且轮廓无法识别：请确认模特姿势是否与参考图基本一致"

    if M_final is None:
        return None, "无法计算对齐路径，请检查图片"

    # 5. 锁定纵横比 (Aspect Ratio Lock)
    s_x = np.sqrt(M_final[0, 0]**2 + M_final[0, 1]**2)
    s_y = np.sqrt(M_final[1, 0]**2 + M_final[1, 1]**2)
    
    avg_scale = (s_x + s_y) / 2.0
    
    rotation_angle = np.arctan2(M_final[1, 0], M_final[0, 0])
    M_final[0, 0] = avg_scale * np.cos(rotation_angle)
    M_final[0, 1] = -avg_scale * np.sin(rotation_angle)
    M_final[1, 0] = avg_scale * np.sin(rotation_angle)
    M_final[1, 1] = avg_scale * np.cos(rotation_angle)

    # 6. 应用最终变换（在无损的高清原图上切图！）
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
                    
                    with st.spinner(f"正在智能对齐 {org_file.name}..."):
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
