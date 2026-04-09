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
    pil_img.save(buf, format='JPEG', quality=100, dpi=(300, 300))
    return buf.getvalue()

# ================= 文件名智能匹配逻辑 =================
def find_matching_reference(org_filename, ref_files):
    org_base = os.path.splitext(org_filename)[0]
    for ref in ref_files:
        ref_base = os.path.splitext(ref.name)[0]
        if org_base == ref_base:
            return ref
    for ref in ref_files:
        ref_base = os.path.splitext(ref.name)[0]
        if org_base.endswith(ref_base):
            return ref
    return None

# ================= 核心算法：双轨制高清对齐 =================
def align_and_crop(org_img_highres, ref_img_highres):
    h_org, w_org = org_img_highres.shape[:2]
    h_ref, w_ref = ref_img_highres.shape[:2]

    max_calc_size = 800
    scale_org = 1.0
    if max(h_org, w_org) > max_calc_size:
        scale_org = max(h_org, w_org) / max_calc_size
        org_img_small = cv2.resize(org_img_highres, (int(w_org / scale_org), int(h_org / scale_org)))
    else:
        org_img_small = org_img_highres.copy()

    scale_ref = 1.0
    if max(h_ref, w_ref) > max_calc_size:
        scale_ref = max(h_ref, w_ref) / max_calc_size
        ref_img_small = cv2.resize(ref_img_highres, (int(w_ref / scale_ref), int(h_ref / scale_ref)))
    else:
        ref_img_small = ref_img_highres.copy()

    gray_org = cv2.cvtColor(org_img_small, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img_small, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_org, None)

    if des2 is None or len(kp2) < 10:
        return None, "原图特征点太少，无法对齐"

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return None, "匹配点不足，图片差异过大"

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    src_pts = src_pts * scale_ref  
    dst_pts = dst_pts * scale_org  

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None: return None, "无法计算透视变换矩阵"

    pts_ref = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
    
    try:
        dst_corners = cv2.perspectiveTransform(pts_ref, M)
    except:
        return None, "数学变换失败，图片变形过大"

    dst_pts_affine = np.float32([dst_corners[0][0], dst_corners[1][0], dst_corners[2][0], dst_corners[3][0]])
    ref_pts_affine = np.float32([[0, 0], [0, h_ref], [w_ref, h_ref], [w_ref, 0]])

    affine_matrix = cv2.getPerspectiveTransform(dst_pts_affine, ref_pts_affine)
    
    result_highres = cv2.warpPerspective(org_img_highres, affine_matrix, (w_ref, h_ref), 
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result_highres, "成功"


# ================= 网页界面 (Streamlit) =================
st.set_page_config(page_title="标准需求自动截图", page_icon="📸", layout="wide")

st.title("📸 标准需求自动截图")
st.markdown("**支持智能匹配：** 原图和参考图依靠文件名自动匹配（原图 `21.jpg` 会自动使用 `1.jpg` 作参考）。导出结果为 **参考图的命名**，画质为 **100% 质量、300 DPI 的 JPG**。")
st.divider()

col1, col2 = st.columns(2)

with col1:
    org_files = st.file_uploader("1️⃣ 上传【修好的高清原图-带头版】(可多选)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key="org")

with col2:
    ref_files = st.file_uploader("2️⃣ 上传【参考的拍图模板图-截头后】(可多选)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key="ref")

if org_files and ref_files:
    st.divider()
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        start_btn = st.button(f"🚀 开始无损比对处理 ({len(org_files)}张)", use_container_width=True, type="primary")

    if start_btn:
        zip_buffer = BytesIO()
        success_count = 0
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, org_file in enumerate(org_files):
                st.write(f"**正在处理: {org_file.name}**")
                matched_ref_file = find_matching_reference(org_file.name, ref_files)
                
                if not matched_ref_file:
                    st.warning(f"⚠️ 跳过：未找到对应的参考图。")
                    st.write("---")
                    continue
                
                try:
                    org_img = load_raw_image(org_file)
                    matched_ref_file.seek(0) 
                    ref_img = load_raw_image(matched_ref_file)
                    
                    result_img, msg = align_and_crop(org_img, ref_img)

                    if result_img is not None:
                        ref_display = cv2.resize(ref_img, (0,0), fx=0.3, fy=0.3)
                        res_display = cv2.resize(result_img, (0,0), fx=0.3, fy=0.3)
                        
                        ref_rgb = cv2.cvtColor(ref_display, cv2.COLOR_BGR2RGB)
                        result_rgb = cv2.cvtColor(res_display, cv2.COLOR_BGR2RGB)
                        
                        d_col1, d_col2, d_col3, d_col4 = st.columns([1, 1, 2, 2])
                        with d_col1: st.image(ref_rgb, caption=f"参考", use_container_width=True)
                        with d_col2: st.image(result_rgb, caption=f"结果", use_container_width=True)
                        with d_col3: st.success(f"✅ 处理成功 (300 DPI)")
                        
                        # ================= 命名逻辑修改 =================
                        # 提取【参考图】的文件名（去掉后缀）
                        ref_base_name = os.path.splitext(matched_ref_file.name)[0]
                        # 强制指定新文件名为：参考图文件名.jpg (例如：1.jpg)
                        final_filename = f"{ref_base_name}.jpg"
                        
                        result_bytes = convert_cv_to_bytes(result_img)
                        zip_file.writestr(final_filename, result_bytes)
                        success_count += 1
                    else:
                        st.error(f"❌ 失败：{msg}")
                        
                except Exception as e:
                    st.error(f"⚠️ 错误: {str(e)}")
                
                st.write("---")
        
        if success_count > 0:
            st.download_button(
                label=f"📦 下载 {success_count} 张处理好的图片 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="aligned_highres_images.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
