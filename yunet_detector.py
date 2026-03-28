from pathlib import Path
import urllib.request

import cv2 as cv
import numpy as np


def ensure_yunet_model(model_path: Path) -> None:
    if model_path.exists() and model_path.stat().st_size > 0:
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceTracking/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
        data = r.read()

    if len(data) < 100000:
        raise SystemExit("YuNet 模型下载失败，文件大小异常。请关闭代理后重试。")
    model_path.write_bytes(data)


class YuNetDetector:
    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (320, 240),
        score_threshold: float = 0.88,
        nms_threshold: float = 0.3,
        top_k: int = 1000,
    ) -> None:
        self._detector = cv.FaceDetectorYN_create(
            model_path,
            "",
            input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def set_input_size(self, size: tuple[int, int]) -> None:
        self._detector.setInputSize(size)

    def infer(self, image: np.ndarray) -> np.ndarray:
        _, faces = self._detector.detect(image)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces
