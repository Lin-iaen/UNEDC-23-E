import cv2
import math
import numpy as np
from typing import Optional, TypedDict


class QuadInfo(TypedDict):
    pts: np.ndarray
    area: float
    cx: float
    cy: float


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """将四边形点排序为 [左上, 右上, 右下, 左下]。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def process_init_mode(frame: np.ndarray):
    """INIT 模式图像处理桩函数。

    返回：
    - annotated_frame: 画了边框和角点标注的图像
    - corners: 若检测到四边形，返回 4 个角点 [(x1,y1)...]；否则返回 []
    """
    annotated = frame.copy()

    # 1) 预处理：灰度 + 高斯滤波，降低噪声干扰
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # 2) 边缘检测：Canny 提取主要边缘
    edges = cv2.Canny(blur, 40, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3) 轮廓提取：保留所有层级，便于识别内外两层胶带。
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quads: list[QuadInfo] = []

    # 4) 候选四边形筛选：面积阈值 + 多边形拟合，并记录面积与质心。
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 1000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue

        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        pts = approx.reshape(4, 2).astype(np.float32)

        quads.append({"pts": pts, "area": float(area), "cx": cx, "cy": cy})

    quads.sort(key=lambda q: q["area"], reverse=True)

    corners = []
    best_ordered: Optional[np.ndarray] = None

    # 核心归一化：在前 5 个候选四边形中做双重循环，寻找稳定的内外双层配对。
    if len(quads) >= 2:
        search_quads = quads[:5]
        matched_pair = False

        for i in range(len(search_quads) - 1):
            q_outer = search_quads[i]
            for j in range(i + 1, len(search_quads)):
                q_inner = search_quads[j]

                cx1, cy1 = float(q_outer["cx"]), float(q_outer["cy"])
                cx2, cy2 = float(q_inner["cx"]), float(q_inner["cy"])
                dist = math.hypot(cx1 - cx2, cy1 - cy2)

                area_outer = float(q_outer["area"])
                area_inner = float(q_inner["area"])
                if area_inner <= 0:
                    continue
                area_ratio = area_outer / area_inner

                if dist < 60.0 and 1.2 < area_ratio < 4.0:
                    p_outer = _order_quad_points(np.asarray(q_outer["pts"], dtype=np.float32))
                    p_inner = _order_quad_points(np.asarray(q_inner["pts"], dtype=np.float32))

                    best_ordered = (p_outer + p_inner) / 2.0

                    # 蓝色绘制两层原始框，绿色粗线绘制平均骨架框。
                    cv2.drawContours(annotated, [p_outer.astype(np.int32)], -1, (255, 0, 0), 2)
                    cv2.drawContours(annotated, [p_inner.astype(np.int32)], -1, (255, 0, 0), 2)
                    cv2.drawContours(annotated, [best_ordered.astype(np.int32)], -1, (0, 255, 0), 5)

                    matched_pair = True
                    break

            if matched_pair:
                break

    # 未形成双层时，降级使用面积最大的一层。
    if best_ordered is None and len(quads) >= 1:
        best_ordered = _order_quad_points(np.asarray(quads[0]["pts"], dtype=np.float32))
        cv2.drawContours(annotated, [best_ordered.astype(np.int32)], -1, (0, 255, 0), 3)

    if best_ordered is not None:
        corners = [(float(x), float(y)) for x, y in best_ordered]

        for i, (x, y) in enumerate(corners):
            xi, yi = int(x), int(y)
            cv2.circle(annotated, (xi, yi), 6, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"P{i}({xi},{yi})",
                (xi + 10, yi - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    # 左上角叠加缩小版边缘图，辅助调参。
    # edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # edges_small = cv2.resize(edges_bgr, (160, 120))
    # annotated[0:120, 0:160] = edges_small

    return annotated, corners

def process(frame: np.ndarray) -> np.ndarray:
    """
    状态机：阶段一（建图模式）
    核心任务：寻找电工胶带围成的四边形，并提取四个顶点
    """
    annotated, _ = process_init_mode(frame)
    return annotated


def process_tracking_mode(
    frame: np.ndarray,
    M: np.ndarray,
    current_target: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, Optional[tuple[float, float]]]:
    """TRACKING 模式图像处理。

    输入：
    - frame: 当前低曝光图像帧（BGR）
    - M: INIT 阶段计算得到的 3x3 透视矩阵
    - current_target: 完美画布坐标系中的当前目标点 (X, Y)

    返回：
    - annotated: 带可视化标注的图像
    - mapped_xy: 映射到完美画布坐标的 (X, Y)，未检测到激光点时为 None
    """
    annotated = frame.copy()

    # 若给定目标点，先做逆透视映射，绘制原图像素系中的“诱饵”大圆。
    if current_target is not None:
        try:
            M_inv = np.linalg.inv(M)
            target_pt = np.array([[[current_target[0], current_target[1]]]], dtype=np.float32)
            target_uv = cv2.perspectiveTransform(target_pt, M_inv)
            tu = int(round(float(target_uv[0, 0, 0])))
            tv = int(round(float(target_uv[0, 0, 1])))
            cv2.circle(annotated, (tu, tv), 15, (0, 255, 255), 3)
        except np.linalg.LinAlgError:
            # M 不可逆时跳过诱饵绘制，避免中断追踪主流程。
            pass

    # 1) LAB 空间提取红激光：结合发白中心与红色光晕两类响应。
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    _ = b_ch  # 保留拆分完整性，便于后续按需扩展阈值策略。

    center_mask = (l_ch > 180) & (a_ch > 130)
    halo_mask = (l_ch > 60) & (a_ch > 135)
    mask = center_mask | halo_mask
    mask_u8 = (mask.astype(np.uint8)) * 255

    # 先闭后开：修复光斑内部断裂，再去除小孤立噪声。
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # 2) 在掩膜中提取最大连通域（轮廓）并计算质心。
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return annotated, None

    # ====== 带通滤波器 ======
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤掉极小的幽灵噪点 (Area < 5) 
        # 过滤掉巨大的红衣服、红瓶盖 (Area > 300)
        # 这个范围你可以根据实际激光点大小微调
        if 5.0 <= area <= 300.0:
            valid_contours.append(cnt)

    # 如果过滤完之后，一个合格的轮廓都没了，说明激光确实不在画面里
    if not valid_contours:
        return annotated, None

    # 在“合格”的轮廓里，挑一个最大的（应对激光晕开的情况）
    largest = max(valid_contours, key=cv2.contourArea)
    # ============================================

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return annotated, None

    u = float(m["m10"] / m["m00"])
    v = float(m["m01"] / m["m00"])

    # 3) 透视跃迁：将像素点 (u, v) 映射到完美画布坐标 (X, Y)。
    src_pt = np.array([[[u, v]]], dtype=np.float32)
    dst_pt = cv2.perspectiveTransform(src_pt, M)
    X = float(dst_pt[0, 0, 0])
    Y = float(dst_pt[0, 0, 1])

    # 可视化：十字准星 + 像素坐标与映射坐标。
    ui, vi = int(round(u)), int(round(v))
    cross_len = 12
    cv2.line(annotated, (ui - cross_len, vi), (ui + cross_len, vi), (0, 255, 0), 2)
    cv2.line(annotated, (ui, vi - cross_len), (ui, vi + cross_len), (0, 255, 0), 2)
    cv2.circle(annotated, (ui, vi), 6, (0, 0, 255), 1)

    cv2.putText(
        annotated,
        f"pix=({ui},{vi})",
        (ui + 10, vi - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        annotated,
        f"map=({X:.1f},{Y:.1f})",
        (ui + 10, vi - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    return annotated, (X, Y)