import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from yunet_detector import YuNetDetector, ensure_yunet_model


def _open_camera(preferred_index: int | None) -> cv2.VideoCapture:
    indices = []
    if preferred_index is not None:
        indices.append(preferred_index)
    indices.extend([0, 1, 2, 3])

    tried: list[int] = []
    for idx in indices:
        if idx in tried:
            continue
        tried.append(idx)

        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        # 读一帧确认真的能出画面（有些“虚拟摄像头”会打开但没图）
        ok, _ = cap.read()
        if ok:
            return cap

        cap.release()

    raise SystemExit(
        "打不开可用摄像头或读不到画面。\n"
        "你可以试试：python .\\main.py --camera 0（或 1/2/3）。"
    )


def _visualize(
    frame: np.ndarray,
    smooth_boxes: list[tuple[float, float, float, float]],
    fps: float,
    mode_text: str,
) -> np.ndarray:
    out = frame.copy()
    
    # 绘制所有人脸框
    for i, (sx, sy, sw, sh) in enumerate(smooth_boxes):
        x1, y1 = int(sx), int(sy)
        x2, y2 = int(sx + sw), int(sy + sh)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 可选：标注人脸编号
        cv2.putText(
            out,
            f"Face {i+1}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"FPS: {fps:.1f}  (press Q to quit)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        mode_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="YuNet realtime face box tracking")
    parser.add_argument("--camera", type=int, default=None, help="摄像头序号：0/1/2/3...")
    parser.add_argument("--score", type=float, default=0.88, help="置信度阈值（高=更严格）")
    parser.add_argument("--nms", type=float, default=0.3, help="NMS 阈值")
    parser.add_argument("--top_k", type=int, default=1000, help="NMS 前候选框数量")
    parser.add_argument("--detect_every", type=int, default=2, help="每 N 帧跑一次检测（越大越快）")
    parser.add_argument("--infer_width", type=int, default=320, help="检测时缩放宽度（越小越快）")
    args = parser.parse_args()

    cap = _open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    # 降低分辨率以提升速度（轻量级）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model_path = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
    ensure_yunet_model(model_path)

    detector = YuNetDetector(
        model_path=str(model_path),
        input_size=(320, 240),
        score_threshold=args.score,
        nms_threshold=args.nms,
        top_k=args.top_k,
    )

    last_t = time.time()
    fps = 0.0

    # 轻量平滑，减少框抖动
    smooth_boxes: list[tuple[float, float, float, float]] = []
    alpha = 0.35  # 越小越平滑，越大越跟手
    frame_idx = 0
    
    # 预先计算检测频率，避免每次循环重复计算
    detect_every = max(1, args.detect_every)
    consecutive_failures = 0
    max_failures = 5  # 连续失败次数阈值

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"警告：摄像头连续读取失败 {consecutive_failures} 次，退出循环")
                    break
                continue
            
            consecutive_failures = 0  # 重置失败计数
            frame_idx += 1
            h, w = frame.shape[:2]
            
            # 计算推理尺寸，确保不会出现除零错误
            infer_w = max(160, min(args.infer_width, w))
            infer_h = max(120, int(h * (infer_w / w)) if infer_w > 0 else 120)

            # 大幅提速：在小图上检测，并且不是每帧都检测
            if frame_idx % detect_every == 0:
                small = cv2.resize(frame, (infer_w, infer_h))
                detector.set_input_size((infer_w, infer_h))
                faces = detector.infer(small)
                
                # 安全缩放，避免除零
                scale_x = w / infer_w if infer_w > 0 else 1.0
                scale_y = h / infer_h if infer_h > 0 else 1.0
                
                # 处理所有检测到的人脸
                current_boxes: list[tuple[float, float, float, float]] = []
                for face in faces:
                    sx, sy, sw, sh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
                    bx, by, bw, bh = sx * scale_x, sy * scale_y, sw * scale_x, sh * scale_y
                    current_boxes.append((bx, by, bw, bh))
                
                # 更新平滑框（与当前检测结果匹配）
                if len(current_boxes) == 0:
                    smooth_boxes = []
                else:
                    new_smooth_boxes: list[tuple[float, float, float, float]] = []
                    for curr_box in current_boxes:
                        # 查找最近的前一帧框（简单的跟踪策略）
                        matched_idx = None
                        if len(smooth_boxes) > 0:
                            # 找到距离最近的框
                            min_dist = float('inf')
                            for i, prev_box in enumerate(smooth_boxes):
                                px, py, pw, ph = prev_box
                                cx, cy, cw, ch = curr_box
                                # 使用中心点距离
                                dist = (px + pw/2 - cx - cw/2)**2 + (py + ph/2 - cy - ch/2)**2
                                if dist < min_dist:
                                    min_dist = dist
                                    matched_idx = i
                        
                        if matched_idx is not None:
                            # 应用平滑
                            px, py, pw, ph = smooth_boxes[matched_idx]
                            cx, cy, cw, ch = curr_box
                            smoothed = (
                                (1 - alpha) * px + alpha * cx,
                                (1 - alpha) * py + alpha * cy,
                                (1 - alpha) * pw + alpha * cw,
                                (1 - alpha) * ph + alpha * ch,
                            )
                            new_smooth_boxes.append(smoothed)
                        else:
                            # 新检测到的人脸，直接初始化
                            new_smooth_boxes.append(curr_box)
                    
                    smooth_boxes = new_smooth_boxes

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

            mode_text = (
                f"Mode: YuNet s>={args.score:.2f} nms={args.nms:.2f} "
                f"k={args.top_k} every={detect_every} w={infer_w} faces={len(smooth_boxes)}"
            )
            frame = _visualize(frame, smooth_boxes, fps, mode_text)

            cv2.imshow("Face Box Tracking (Lightweight)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # q / ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
