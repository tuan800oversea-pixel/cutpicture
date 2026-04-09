import streamlit as st
import cv2
import numpy as np
from io import BytesIO

# ================= 核心算法部分 (基本保持原样) =================
def align_and_crop(org_img, ref_img):
    """
    通过 SIFT 特征匹配将原图对齐到参考图的视角
    """
    gray_org = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_ref, None)  # 参考图
    kp2, des2 = sift.detectAndCompute(gray_org, None)  # 原图

    if des2 is None or len(kp2) < 10:
        return None, "原图特征点太少，无法对齐，请换一张图片重试。"

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return None, "匹配点不足，原图和参考图内容差异过大，请确保两张图拍的是相同的主体。"

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h_ref, w_ref = ref_img.shape[:2]
    pts_ref = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)

    try:
        dst_corners = cv2.perspectiveTransform(pts_ref, M)
    except:
        return None, "数学变换失败，图片可能变形过大。"

    dst_pts_affine = np.float32([dst_corners[0][0], dst_corners[1][0], dst_corners[2][0], dst_corners[3][0]])
    ref_pts_affine = np.float32([[0, 0], [0, h_ref], [w_ref, h_ref], [w_ref, 0]])

    affine_matrix = cv2.getPerspectiveTransform(dst_pts_affine, ref_pts_affine)
    result = cv2.warpPerspective(org_img, affine_matrix, (w_ref, h_ref), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

    return result, "成功"

# ================= 辅助函数：处理上传的图片 =================
def load_uploaded_image(uploaded_file):
    # 将前端上传的文件转换为 OpenCV 可以处理的格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def convert_cv_to_bytes(cv_img, ext='.jpg'):
    # 将 OpenCV 处理后的图片转换回字节流，用于下载按钮
    res, img_encode = cv2.imencode(ext, cv_img)
    return BytesIO(img_encode.tobytes())

# ================= 网页界面 (Streamlit) =================
# 1. 页面基本设置
st.set_page_config(page_title="图片视角对齐工具", page_icon="📸", layout="wide")

# 2. 标题和使用说明
st.title("📸 图片特征构图与视角对齐克隆工具")
st.markdown("""
**欢迎使用！** 本工具可以提取参考图的视角和构图，并将你的原图裁切、拉伸成与之完全一致的画面。
*无需人脸识别，完全基于图像特征（如花纹、边缘）进行物理对齐。*

**👉 使用要求：**
1. **原图** 和 **参考图** 必须包含相同的物体或场景（否则找不到特征点）。
2. 原图的视野最好比参考图大，这样裁切后才不会出现白边。
""")

st.divider() # 分割线

# 3. 上传区域 (分两列显示)
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ 上传你需要处理的【原图】")
    st.info("这是你想要改变视角的那张图。")
    org_file = st.file_uploader("选择原图 (JPG/PNG)", type=['png', 'jpg', 'jpeg'], key="org")

with col2:
    st.subheader("2️⃣ 上传完美的【参考图】")
    st.info("我们将提取这张图的构图视角。")
    ref_file = st.file_uploader("选择参考图 (JPG/PNG)", type=['png', 'jpg', 'jpeg'], key="ref")

# 4. 如果两张图都上传了，显示预览并开始处理
if org_file and ref_file:
    # 预览上传的图片
    st.markdown("### 👀 预览输入")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.image(org_file, caption="原图预览", use_container_width=True)
    with p_col2:
        st.image(ref_file, caption="参考图预览", use_container_width=True)
        
    st.divider()
    
    # 居中的处理按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        start_btn = st.button("🚀 开始对齐处理", use_container_width=True, type="primary")

    if start_btn:
        with st.spinner("正在进行 SIFT 特征匹配和透视计算，请稍候..."):
            # 转换格式
            org_img = load_uploaded_image(org_file)
            ref_img = load_uploaded_image(ref_file)

            # 运行核心算法
            result_img, msg = align_and_crop(org_img, ref_img)

            if result_img is not None:
                st.success("✅ 对齐成功！")
                
                # 展示结果：Streamlit 展示需要 RGB 格式，OpenCV 默认是 BGR，所以要转换一下
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # 为了直观对比，我们把结果和参考图放在一起看
                st.markdown("### 🏆 对齐结果对比")
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.image(ref_file, caption="你的参考图 (目标)", use_container_width=True)
                with r_col2:
                    st.image(result_rgb, caption="✨ 处理完成的原图 (结果)", use_container_width=True)
                
                # 提供下载按钮
                result_bytes = convert_cv_to_bytes(result_img)
                st.download_button(
                    label="💾 下载对齐后的图片",
                    data=result_bytes,
                    file_name="aligned_output.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            else:
                st.error(f"❌ 处理失败：{msg}")
