import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import os
import zipfile

# ================= 图片加载与高清导出 =================
def load_raw_image(uploaded_file):
    # 直接读取原始尺寸，不做任何压缩
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def convert_cv_to_bytes(cv_img, ext='.jpg'):
    # 强制以最高质量/无损方式导出字节流
    ext_lower = ext.lower()
    if ext_lower in ['.jpg', '.jpeg']:
        # JPG 设置为 100% 质量 (默认通常是 95)
        res, img_encode = cv2.imencode(ext, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif ext_lower == '.png':
        # PNG 设置为 0 压缩 (最高质量，文件最大)
        res, img_encode = cv2.imencode(ext, cv_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    else:
        res, img_encode = cv2.imencode(ext, cv_img)
    return img_encode.tobytes()

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
    # 1. 记录原始高清尺寸
    h_org, w_org = org_img_highres.shape[:2]
    h_ref, w_ref = ref_img_highres.shape[:2]

    # 2. 为了防止服务器崩溃，生成用于“算数学题”的小图 (最长边800)
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

    # 3. 在小图上跑 SIFT 特征点提取
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

    # 4. 获取小图上的匹配坐标，并【乘回缩放比例】还原到高清坐标系！
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    src_pts = src_pts * scale_ref  # 还原为参考原图的真实像素位置
    dst_pts = dst_pts * scale_org  # 还原为上传原图的真实像素位置

    # 5. 在高清坐标下计算透视矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None: return None, "无法计算透视变换矩阵"

    # 使用高清参考图的四个角
    pts_ref = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
    
    try:
        dst_corners = cv2.perspectiveTransform(pts_ref, M)
    except:
        return None, "数学变换失败，图片变形过大"

    dst_pts_affine = np.float32([dst_corners[0][0], dst_corners[1][0], dst_corners[2][0], dst_corners[3][0]])
    ref_pts_affine = np.float32([[0, 0], [0, h_ref], [w_ref, h_ref], [w_ref, 0]])

    affine_matrix = cv2.getPerspectiveTransform(dst_pts_affine, ref_pts_affine)
    
    # 6. 【高光时刻】拿算好的矩阵，直接去裁切最高清的原始图片！
    result_highres = cv2.warpPerspective(org_img_highres, affine_matrix, (w_ref, h_ref), 
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return result_highres, "成功"


# ================= 网页界面 (Streamlit) =================
st.set_page_config(page_title="无损图片对齐工具", page_icon="📸", layout="wide")

st.title("📸 标准需求自动截图")
st.markdown("**支持智能匹配：** 原图和参考图依靠文件名自动匹配（原图 `21.jpg` 会自动使用 `1.jpg` 作参考）。导出结果为原图 100% 尺寸与画质。")
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
                    # 加载未经任何压缩的图片
                    org_img = load_raw_image(org_file)
                    matched_ref_file.seek(0) 
                    ref_img = load_raw_image(matched_ref_file)
                    
                    result_img, msg = align_and_crop(org_img, ref_img)

                    if result_img is not None:
                        # 网页展示用缩略图即可（避免网页卡顿），但下载存的是上面的 result_img 高清图！
                        ref_display = cv2.resize(ref_img, (0,0), fx=0.3, fy=0.3)
                        res_display = cv2.resize(result_img, (0,0), fx=0.3, fy=0.3)
                        
                        ref_rgb = cv2.cvtColor(ref_display, cv2.COLOR_BGR2RGB)
                        result_rgb = cv2.cvtColor(res_display, cv2.COLOR_BGR2RGB)
                        
                        d_col1, d_col2, d_col3, d_col4 = st.columns([1, 1, 2, 2])
                        with d_col1: st.image(ref_rgb, caption=f"参考", use_container_width=True)
                        with d_col2: st.image(result_rgb, caption=f"结果", use_container_width=True)
                        with d_col3: st.success(f"✅ 处理成功 (保留原画质)")
                        
                        ext = os.path.splitext(org_file.name)[1]
                        result_bytes = convert_cv_to_bytes(result_img, ext=ext)
                        zip_file.writestr(f"Aligned_{org_file.name}", result_bytes)
                        success_count += 1
                    else:
                        st.error(f"❌ 失败：{msg}")
                        
                except Exception as e:
                    st.error(f"⚠️ 错误: {str(e)}")
                
                st.write("---")
        
        if success_count > 0:
            st.download_button(
                label=f"📦 下载 {success_count} 张无损高清结果 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="aligned_highres_images.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
