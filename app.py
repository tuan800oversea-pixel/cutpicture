import streamlit as st
import cv2
import numpy as np
from io import BytesIO

# ================= 新增：防崩溃的图片缩放机制 =================
def resize_image(img, max_size=1200):
    """
    智能缩放：如果图片最长边超过 max_size，则等比例缩小。
    大幅减少内存消耗，防止云端服务器崩溃。
    """
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def load_uploaded_image(uploaded_file):
    # 将前端上传的文件转换为 OpenCV 格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return resize_image(img) # 加载的同时进行安全缩放

def convert_cv_to_bytes(cv_img, ext='.jpg'):
    # 将 OpenCV 处理后的图片转换回字节流
    res, img_encode = cv2.imencode(ext, cv_img)
    return BytesIO(img_encode.tobytes())

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

st.title("📸 图片特征构图与视角对齐克隆工具")
st.markdown("""
**欢迎使用！** 本工具可以提取参考图的视角和构图，并将你的原图裁切、拉伸成与之完全一致的画面。
*无需人脸识别，完全基于图像特征（如花纹、边缘）进行物理对齐。*

**👉 使用要求：**
1. **原图** 和 **参考图** 必须包含相同的物体或场景。
2. 原图的视野最好比参考图大，这样裁切后才不会出现白边。
""")

st.divider()

# 修改为多文件和单文件组合
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ 上传需要处理的【原图】(可多选)")
    st.info("支持一次性拖拽或选择多张你想改变视角的图片。")
    # accept_multiple_files=True 开启批量上传
    org_files = st.file_uploader("选择原图 (JPG/PNG)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key="org")

with col2:
    st.subheader("2️⃣ 上传完美的【参考图】(单张)")
    st.info("我们将提取这张图的构图视角作为标准。")
    ref_file = st.file_uploader("选择参考图 (JPG/PNG)", type=['png', 'jpg', 'jpeg'], key="ref")

# 如果上传了原图(列表不为空) 并且 上传了参考图
if org_files and ref_file:
    st.divider()
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        start_btn = st.button(f"🚀 开始批量处理 ({len(org_files)}张)", use_container_width=True, type="primary")

    if start_btn:
        st.markdown("### 🏆 处理结果")
        
        # 加载参考图 (只需要加载一次)
        with st.spinner("正在加载参考图..."):
            ref_img = load_uploaded_image(ref_file)
            st.image(ref_file, caption="🎯 你的标准参考图", width=300)
            
        st.write("---")

        # 循环处理每一张原图
        for idx, org_file in enumerate(org_files):
            with st.container():
                st.write(f"**处理进度: {idx + 1} / {len(org_files)} - 文件名: {org_file.name}**")
                
                try:
                    # 加入容错机制，单张失败不会导致全盘崩溃
                    org_img = load_uploaded_image(org_file)
                    result_img, msg = align_and_crop(org_img, ref_img)

                    if result_img is not None:
                        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        
                        r_col1, r_col2 = st.columns([1, 2])
                        with r_col1:
                            st.image(org_file, caption="原图预览", use_container_width=True)
                        with r_col2:
                            st.image(result_rgb, caption="✨ 处理完成", use_container_width=True)
                            
                            # 下载按钮
                            result_bytes = convert_cv_to_bytes(result_img)
                            st.download_button(
                                label=f"💾 下载这图",
                                data=result_bytes,
                                file_name=f"aligned_{org_file.name}",
                                mime="image/jpeg",
                                key=f"dl_{idx}" # key必须唯一
                            )
                        st.success("✅ 成功")
                    else:
                        st.warning(f"❌ 失败：{msg}")
                        
                except Exception as e:
                    # 如果发生未知错误，捕获并在网页上显示，而不是让网页崩溃
                    st.error(f"⚠️ 处理 {org_file.name} 时发生系统错误: {str(e)}")
                
                st.write("---")
