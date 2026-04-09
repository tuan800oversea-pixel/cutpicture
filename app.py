import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import os
import zipfile

# ================= 防崩溃与图片加载 =================
def resize_image(img, max_size=800):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def load_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return resize_image(img)

def convert_cv_to_bytes(cv_img, ext='.jpg'):
    res, img_encode = cv2.imencode(ext, cv_img)
    return img_encode.tobytes()

# ================= 文件名智能匹配逻辑 =================
def find_matching_reference(org_filename, ref_files):
    """
    根据原图文件名寻找对应的参考图。
    规则1：完全同名 (不看后缀)
    规则2：原图名字的结尾包含参考图的名字 (例如：21 匹配 1)
    """
    org_base = os.path.splitext(org_filename)[0]
    
    # 优先尝试完全匹配
    for ref in ref_files:
        ref_base = os.path.splitext(ref.name)[0]
        if org_base == ref_base:
            return ref
            
    # 尝试尾号/后缀匹配
    for ref in ref_files:
        ref_base = os.path.splitext(ref.name)[0]
        if org_base.endswith(ref_base):
            return ref
            
    return None

# ================= 核心算法部分 =================
def align_and_crop(org_img, ref_img):
    gray_org = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

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

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        return None, "无法计算透视变换矩阵"

    h_ref, w_ref = ref_img.shape[:2]
    pts_ref = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)

    try:
        dst_corners = cv2.perspectiveTransform(pts_ref, M)
    except:
        return None, "数学变换失败，图片可能变形过大"

    dst_pts_affine = np.float32([dst_corners[0][0], dst_corners[1][0], dst_corners[2][0], dst_corners[3][0]])
    ref_pts_affine = np.float32([[0, 0], [0, h_ref], [w_ref, h_ref], [w_ref, 0]])

    affine_matrix = cv2.getPerspectiveTransform(dst_pts_affine, ref_pts_affine)
    result = cv2.warpPerspective(org_img, affine_matrix, (w_ref, h_ref), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    return result, "成功"


# ================= 网页界面 (Streamlit) =================
st.set_page_config(page_title="图片视角对齐工具", page_icon="📸", layout="wide")

st.title("📸 标准需求自动截图)")
st.markdown("""
**命名规则提示：**
原图和参考图依靠**文件名**进行自动匹配。你可以让它们**完全同名**，或者原图名字**以参考图名字结尾**（例如：原图 `21.jpg` 会自动使用 `1.jpg` 作为参考图去裁切）。
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ 上传【原图】(可多选)")
    org_files = st.file_uploader("选择需要裁切的原图", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key="org")

with col2:
    st.subheader("2️⃣ 上传【参考的拍图模板图】(可多选)")
    ref_files = st.file_uploader("选择作为标准的参考图", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key="ref")

# 如果两边都上传了文件
if org_files and ref_files:
    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        start_btn = st.button(f"🚀 开始批量比对处理 ({len(org_files)}张)", use_container_width=True, type="primary")

    if start_btn:
        st.markdown("### 🏆 处理结果")
        
        # 用于打包 ZIP 文件的内存缓冲区
        zip_buffer = BytesIO()
        success_count = 0
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, org_file in enumerate(org_files):
                st.write(f"**处理: {org_file.name}**")
                
                # 寻找对应的参考图
                matched_ref_file = find_matching_reference(org_file.name, ref_files)
                
                if not matched_ref_file:
                    st.warning(f"⚠️ 跳过：没有找到与 {org_file.name} 匹配的参考图。")
                    st.write("---")
                    continue
                
                try:
                    # 读取图片
                    org_img = load_uploaded_image(org_file)
                    
                    # 每次读取匹配到的参考图（因为参考图可能有多个）
                    matched_ref_file.seek(0) 
                    ref_img = load_uploaded_image(matched_ref_file)
                    
                    # 处理对齐
                    result_img, msg = align_and_crop(org_img, ref_img)

                    if result_img is not None:
                        # 转换颜色格式用于网页显示
                        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        
                        # 显示变小：使用更多的列来挤压图片宽度
                        d_col1, d_col2, d_col3, d_col4 = st.columns([1, 1, 2, 2])
                        with d_col1:
                            st.image(ref_rgb, caption=f"参考标准 ({matched_ref_file.name})", use_container_width=True)
                        with d_col2:
                            st.image(result_rgb, caption=f"裁切结果", use_container_width=True)
                        with d_col3:
                            st.success(f"✅ 对齐成功")
                        
                        # 将成功的结果写入 ZIP 包中
                        result_bytes = convert_cv_to_bytes(result_img, ext=os.path.splitext(org_file.name)[1])
                        zip_file.writestr(f"aligned_{org_file.name}", result_bytes)
                        success_count += 1
                    else:
                        st.error(f"❌ 失败：{msg}")
                        
                except Exception as e:
                    st.error(f"⚠️ 系统错误: {str(e)}")
                
                st.write("---")
        
        # 批量处理结束，如果有成功的图片，显示下载压缩包按钮
        if success_count > 0:
            st.success(f"🎉 全部处理完成！共成功 {success_count} 张。")
            st.download_button(
                label="📦 一键下载所有处理好的图片 (ZIP压缩包)",
                data=zip_buffer.getvalue(),
                file_name="aligned_images.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
