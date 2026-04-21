import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import os
import zipfile
from PIL import Image

# 引入 MediaPipe，用于人体语义关键点检测
import mediapipe as mp

# ================= 图片加载与 300 DPI 处理 (保持原版) =================
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

# ================= 文件名匹配逻辑 (保持原版) =================
def find_matching_reference(org_filename, ref_files):
    org_base = os.path.splitext(org_filename)[0]
    sorted_refs = sorted(ref_files, key=lambda x: len(os.path.splitext(x.name)[0]), reverse=True)
    for ref in sorted_refs:
        ref_base = os.path.splitext(ref.name)[0]
        if org_base == ref_base or org_base.endswith(ref_base):
            return ref
    return None

# ================= 核心算法：人体语义关键点对齐 (究极进化) =================

def get_body_keypoints(img, height_scale=0.5):
    """
    使用 MediaPipePose 检测人体语义关键点。
    为了加快检测速度，将图片缩小进行检测。
    """
    mp_pose = mp.solutions.pose
    h, w = img.shape[:2]
    
    # 缩小计算，提速
    max_calc_size = 600
    scale = max(h, w) / max_calc_size if max(h, w) > max_calc_size else 1.0
    img_small = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    h_small, w_small = img_small.shape[:2]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,  # 使用中等模型，平衡速度和精度
        enable_segmentation=False,
        min_detection_confidence=0.5) as pose:
        
        results = pose.process(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None, "人体结构检测失败，请确认图片中有清晰可辨的人物"

        # 提取相关关键点：
        # 这里提取了腰（hips）和大腿中点（thighs），以对齐泳裤区域
        landmarks = results.pose_landmarks.landmark
        
        # 将相对坐标映射回缩小后的图坐标系
        def to_small_coords(landmark):
            return [int(landmark.x * w_small), int(landmark.y * h_small)]
        
        relevant_indices = [23, 24, 25, 26, 27, 28] # Hips, Knees, Ankles
        points_small = []
        
        # Hips (腰/臀点)
        left_hip = to_small_coords(landmarks[23])
        right_hip = to_small_coords(landmarks[24])
        # Knees (大腿中点/膝盖点)
        left_knee = to_small_coords(landmarks[25])
        right_knee = to_small_coords(landmarks[26])

        # 组合成稳定且相关的点对
        # 1. 腰中点
        hip_center = [(left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2]
        # 2. 大腿中点 (对齐泳裤区域最关键的点对)
        thigh_center = [(left_knee[0] + right_knee[0]) // 2, (left_knee[1] + right_knee[1]) // 2]
        # 3. 大腿长度向量的方向
        thigh_direction = [thigh_center[0] - hip_center[0], thigh_center[1] - hip_center[1]]
        
        points_small = [
            hip_center,       # 点1: 腰中点
            thigh_center,     # 点2: 大腿中点
        ]

        # 计算一个稳定的旋转向量方向
        points_small = np.float32(points_small)
        
        # 将关键点坐标放大回原始高清大图坐标系
        points_highres = points_small * scale
        
        return points_highres, "成功"

def align_and_crop_strict(org_img_highres, ref_img_highres):
    h_org, w_org = org_img_highres.shape[:2]
    h_ref, w_ref = ref_img_highres.shape[:2]

    # 1. 分别提取人体关键点
    mp_points_org, msg_org = get_body_keypoints(org_img_highres)
    mp_points_ref, msg_ref = get_body_keypoints(ref_img_highres)

    if mp_points_org is None:
        return None, f"原图人体结构检测失败：{msg_org}"
    if mp_points_ref is None:
        return None, f"模板图人体结构检测失败：{msg_ref}"

    M_final = None

    # 2. 宏观对齐计算：依靠稳定的点对 (腰中点和大腿中点)
    # 计算一个稳定的相似变换矩阵 (包含旋转、缩放、平移)
    dst_pts = mp_points_org # 源点 (待截图原图)
    src_pts = mp_points_ref # 目标点 (截图模板图)
    
    # 3. 计算最佳相似变换矩阵 (包含旋转、缩放和平移)
    # estimateAffinePartial2D 在点对非常稳定时极其有效
    M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
    
    if M is None:
        # 如果 estimateAffinePartial2D 失败，可能是点对不够多
        # 使用更简单的点对点计算保底
        return None, "无法根据关键点计算变换，两图的模特身形差异可能过大"
        
    # 4. 【核心保留】锁定纵横比 (Aspect Ratio Lock)
    # 确保 avg_scale 极其准确，防止体形被非等比例拉伸
    s_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    s_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
    avg_scale = (s_x + s_y) / 2.0
    rotation_angle = np.arctan2(M[1, 0], M[0, 0])
    
    # 构建等比例相似变换矩阵
    M_final = np.array([
        [avg_scale * np.cos(rotation_angle), -avg_scale * np.sin(rotation_angle), M[0, 2]],
        [avg_scale * np.sin(rotation_angle), avg_scale * np.cos(rotation_angle), M[1, 2]]
    ], dtype=np.float32)

    # 5. 应用最终变换（在高清原图上执行！）
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
                    
                    with st.spinner(f"正在智能对齐 {org_file.name} (初次运行加载人体模型，稍等片刻)..."):
                        # 执行新的关键点对齐
                        res_img, msg = align_and_crop_strict(img_org, img_ref)

                    if res_img is not None:
                        # 转换并打包
                        ref_name = os.path.splitext(matched_ref.name)[0]
                        file_name = f"{ref_name}.jpg"
                        
                        img_bytes = convert_cv_to_bytes(res_img)
                        zip_file.writestr(file_name, img_bytes)
                        
                        # ================= 实时预览展示 =================
                        with st.expander(f"✅ 已处理: {org_file.name} ➔ {file_name}", expanded=True):
                            # 使用原有的排版比例
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
