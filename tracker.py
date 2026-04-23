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
    """INIT 模式图像处理。

    返回：
    - annotated_frame: 画了边框和角点标注的图像
    - corners: 若检测到四边形，返回 4 个角点 [(x1,y1)...]；否则返回 []
    """
    annotated = frame.copy()

    # 1) 预处理：灰度与中值滤波。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # 2) Canny 边缘检测。
    edges = cv2.Canny(blur, 40, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3) 轮廓提取。
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quads: list[QuadInfo] = []

    # 4) 四边形筛选：按面积和拟合边数过滤。
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

    # 在前 5 个候选四边形中寻找内外配对。
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

                    # 蓝色绘制双层框，绿色绘制平均框。
                    cv2.drawContours(annotated, [p_outer.astype(np.int32)], -1, (255, 0, 0), 2)
                    cv2.drawContours(annotated, [p_inner.astype(np.int32)], -1, (255, 0, 0), 2)
                    cv2.drawContours(annotated, [best_ordered.astype(np.int32)], -1, (0, 255, 0), 5)

                    matched_pair = True
                    break

            if matched_pair:
                break

    # 未匹配双层时，退化为单层最大四边形。
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

    # 可开启边缘图叠加以辅助阈值调试。
    # edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # edges_small = cv2.resize(edges_bgr, (160, 120))
    # annotated[0:120, 0:160] = edges_small

    return annotated, corners

def process(frame: np.ndarray) -> np.ndarray:
    """
    兼容接口：INIT 模式处理入口。
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

    # 若存在目标点，先逆透视到原图并绘制目标圈。
    expected_uv: Optional[tuple[float, float]] = None
    if current_target is not None:
        try:
            M_inv = np.linalg.inv(M)
            target_pt = np.array([[[current_target[0], current_target[1]]]], dtype=np.float32)
            target_uv = cv2.perspectiveTransform(target_pt, M_inv)
            tu = int(round(float(target_uv[0, 0, 0])))
            tv = int(round(float(target_uv[0, 0, 1])))
            expected_uv = (float(target_uv[0, 0, 0]), float(target_uv[0, 0, 1]))
            cv2.circle(annotated, (tu, tv), 15, (0, 255, 255), 3)
        except np.linalg.LinAlgError:
            # 矩阵不可逆时跳过绘制，不中断主流程。
            pass

    # 1) 转换到 HSV 并分离通道。
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # 2) 双重掩膜：红晕 + 过曝白洞核心。
    halo_mask = ((h_ch < 10) | (h_ch > 160)) & (s_ch > 100) & (v_ch > 20)
    core_mask = (v_ch > 240) & (s_ch < 80)
    combined_mask = halo_mask | core_mask
    mask_u8 = combined_mask.astype(np.uint8) * 255

    # 3) 形态学：严格先开后闭。
    kernel_open = np.ones((3, 3), dtype=np.uint8)
    kernel_close = np.ones((7, 7), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)

    # 若已有期望目标，优先在目标附近检索 (保持你的原代码不变)
    if expected_uv is not None:
        roi = np.zeros_like(mask_u8)
        ex_u = int(round(expected_uv[0]))
        ex_v = int(round(expected_uv[1]))
        cv2.circle(roi, (ex_u, ex_v), 90, 255, -1)
        roi_mask_u8 = cv2.bitwise_and(mask_u8, roi)
        if cv2.countNonZero(roi_mask_u8) > 0:
            mask_u8 = roi_mask_u8

    # --- Debug 投屏层 ---
    frame_h, frame_w = annotated.shape[:2]

    v_bgr = cv2.cvtColor(v_ch, cv2.COLOR_GRAY2BGR)
    v_small = cv2.resize(v_bgr, (160, 120))
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    mask_small = cv2.resize(mask_bgr, (160, 120))

    annotated[frame_h - 120:frame_h, 0:160] = v_small
    annotated[frame_h - 120:frame_h, frame_w - 160:frame_w] = mask_small

    cv2.putText(
        annotated,
        "Debug: V-Channel",
        (5, frame_h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        annotated,
        "Debug: Final HSV Mask",
        (frame_w - 155, frame_h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
    )

    # 2) 提取轮廓并计算激光质心。
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return annotated, None

    # 面积过滤：放宽下限以接纳黑底上的极小光斑。
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2.0 <= area <= 2500.0:
            valid_contours.append(cnt)

    # 无合格轮廓时返回未检测。
    if not valid_contours:
        return annotated, None

    # 在合格轮廓中按“目标邻近 + 面积”综合打分。
    def _score_contour(cnt: np.ndarray) -> float:
        area = float(cv2.contourArea(cnt))
        if expected_uv is None:
            return area

        m2 = cv2.moments(cnt)
        if m2["m00"] == 0:
            return -1.0

        cu = float(m2["m10"] / m2["m00"])
        cv = float(m2["m01"] / m2["m00"])
        dist = math.hypot(cu - expected_uv[0], cv - expected_uv[1])
        return area - 0.35 * dist

    largest = max(valid_contours, key=_score_contour)

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return annotated, None

    u = float(m["m10"] / m["m00"])
    v = float(m["m01"] / m["m00"])

    # 3) 透视映射：将像素坐标映射到画布坐标。
    src_pt = np.array([[[u, v]]], dtype=np.float32)
    dst_pt = cv2.perspectiveTransform(src_pt, M)
    X = float(dst_pt[0, 0, 0])
    Y = float(dst_pt[0, 0, 1])

    # 可视化：十字准星和坐标文本。
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