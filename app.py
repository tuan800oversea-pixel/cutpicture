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
    for ref in ref_files:
        ref_base = os.path.splitext(ref.name)[0]
        if org_base == ref_base or org_base.endswith("_" + ref_base):
            return ref
    return None

# ================= 核心算法：100% 比例锁定对齐 =================
def align_and_crop_strict(org_img_highres, ref_img_highres):
    h_org, w_org = org_img_highres.shape[:2]
    h_ref, w_ref = ref_img_highres.shape[:2]

    # 1. 预处理：转为中等尺寸进行特征计算，提高精度
    max_calc_size = 1200
    scale_down_org = max(h_org, w_org) / max_calc_size if max(h_org, w_org) > max_calc_size else 1.0
    scale_down_ref = max(h_ref, w_ref) / max_calc_size if max(h_ref, w_ref) > max_calc_size else 1.0

    org_img_small = cv2.resize(org_img_highres, (int(w_org / scale_down_org), int(h_org / scale_down_org)), interpolation=cv2.INTER_AREA)
    ref_img_small = cv2.resize(ref_img_highres, (int(w_ref / scale_down_ref), int(h_ref / scale_down_ref)), interpolation=cv2.INTER_AREA)

    # 2. SIFT 特征提取
    sift = cv2.SIFT_create(nfeatures=2000)
    kp_ref, des_ref = sift.detectAndCompute(cv2.cvtColor(ref_img_small, cv2.COLOR_BGR2GRAY), None)
    kp_org, des_org = sift.detectAndCompute(cv2.cvtColor(org_img_small, cv2.COLOR_BGR2GRAY), None)

    if des_org is None or des_ref is None or len(kp_org) < 10:
        return None, "特征提取失败：图片过于模糊或内容无法匹配"

    # 3. 匹配特征点
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_ref, des_org, k=2)
    # 更加严格的匹配过滤 (0.65)
    good_matches = [m for m, n in matches if m.distance < 0.65 * n.distance]

    if len(good_matches) < 10:
        return None, "匹配点不足：请确认模特姿势是否与参考图基本一致"

    # 4. 坐标映射回原尺寸
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale_down_ref
    dst_pts = np.float32([kp_org[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) * scale_down_org

    # 5. 计算相似变换矩阵（RANSAC 过滤噪声）
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    if M is None:
        return None, "无法计算对齐路径"

    # 6. 【核心优化】锁定纵横比 (Aspect Ratio Lock)
    # M 矩阵的前两列包含了缩放和旋转信息：[[a, b, tx], [c, d, ty]]
    # 计算当前变换的缩放因子
    s_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    s_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
    
    # 强制令 s_x = s_y，取两者的平均值，彻底杜绝拉伸变瘦/变胖
    avg_scale = (s_x + s_y) / 2.0
    
    # 重新归一化矩阵中的旋转部分，并统一应用平均缩放因子
    rotation_angle = np.arctan2(M[1, 0], M[0, 0])
    M[0, 0] = avg_scale * np.cos(rotation_angle)
    M[0, 1] = -avg_scale * np.sin(rotation_angle)
    M[1, 0] = avg_scale * np.sin(rotation_angle)
    M[1, 1] = avg_scale * np.cos(rotation_angle)

    # 7. 应用变换
    result_highres = cv2.warpAffine(org_img_highres, M, (w_ref, h_ref), 
                                    flags=cv2.INTER_LANCZOS4, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(255, 255, 255))
    
    return result_highres, "成功"


# ================= Streamlit UI =================
st.set_page_config(page_title="身材比例锁定-专业截图工具", page_icon="📏", layout="wide")

st.title("📏 身材比例锁定 - 专业截图工具")
st.info("本次更新：加入了 **'纵横比硬锁定'** 算法。无论图片如何对齐，模特的身高、胖瘦比例将 100% 保持原样，彻底解决变形问题。")

col1, col2 = st.columns(2)
with col1:
    org_files = st.file_uploader("1️⃣ 上传高清原图 (待截图)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
with col2:
    ref_files = st.file_uploader("2️⃣ 上传参考模板图 (定坐标)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

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
                    # 读取图片
                    img_org = load_raw_image(org_file)
                    matched_ref.seek(0)
                    img_ref = load_raw_image(matched_ref)
                    
                    # 执行严格对齐
                    res_img, msg = align_and_crop_strict(img_org, img_ref)

                    if res_img is not None:
                        # 转换并打包
                        ref_name = os.path.splitext(matched_ref.name)[0]
                        file_name = f"{ref_name}.jpg"
                        
                        img_bytes = convert_cv_to_bytes(res_img)
                        zip_file.writestr(file_name, img_bytes)
                        
                        # 实时预览
                        with st.expander(f"✅ 已处理: {org_file.name} ➔ {file_name}", expanded=False):
                            preview_res = cv2.resize(res_img, (0,0), fx=0.15, fy=0.15)
                            st.image(cv2.cvtColor(preview_res, cv2.COLOR_BGR2RGB))
                        
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
