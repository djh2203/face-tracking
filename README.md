# 实时人脸框跟踪

一个基于 OpenCV YuNet 的轻量级实时人脸检测工具，支持命令行参数调节，适合快速体验和部署。

## 核心特性

- ✅ **零训练成本**：使用预训练的 YuNet 模型，开箱即用
- ✅ **轻量高效**：低分辨率推理 + 跳帧检测，普通硬件即可流畅运行
- ✅ **灵活配置**：支持置信度、检测频率、平滑系数等参数调节
- ✅ **实时反馈**：显示 FPS 帧率，绿色框标记人脸位置
- ✅ **自动下载**：首次运行自动下载模型文件

---

## 1) 安装 Python

1. 打开微软商店或去官网安装 **Python 3.10+**
2. 安装时勾选 **Add Python to PATH**

安装好后，打开 PowerShell 输入：

```bash
python --version
```

能看到版本号就 OK。

---

## 2) 安装依赖（第一次运行前做一次）

在 PowerShell 里进入这个项目目录：

建议先创建虚拟环境：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

---

## 3) 运行

### 基础运行

```bash
python .\main.py
```

看到窗口后，把脸放到摄像头前，会出现绿色框。按 **Q** 或 **ESC** 退出。

### 命令行参数说明

`main.py` 支持以下参数来调节检测效果：

```bash
python .\main.py [参数选项]
```

| 参数             | 类型  | 默认值 | 说明                                         | 示例                |
| ---------------- | ----- | ------ | -------------------------------------------- | ------------------- |
| `--camera`       | int   | `None` | 指定摄像头编号（0/1/2/3...），多摄像头时使用 | `--camera 1`        |
| `--score`        | float | `0.88` | 置信度阈值，越高越严格（范围 0~1）           | `--score 0.95`      |
| `--nms`          | float | `0.3`  | NMS（非极大值抑制）阈值，去除重叠框          | `--nms 0.4`         |
| `--top_k`        | int   | `1000` | NMS 前保留的候选框数量                       | `--top_k 500`       |
| `--detect_every` | int   | `2`    | 每 N 帧检测一次，越大越快但延迟增加          | `--detect_every 3`  |
| `--infer_width`  | int   | `320`  | 检测时的缩放宽度，越小越快                   | `--infer_width 240` |

#### 使用示例

**使用 1 号摄像头，提高检测严格度：**

```bash
python .\main.py --camera 1 --score 0.95
```

**追求更高帧率（降低精度）：**

```bash
python .\main.py --detect_every 3 --infer_width 240
```

**更严格的框选（减少误检）：**

```bash
python .\main.py --score 0.95 --nms 0.4
```

---

## 4) 技术架构与实现原理

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                   main.py (主入口)                    │
├─────────────────────────────────────────────────────┤
│  1. 解析命令行参数                                    │
│  2. 初始化摄像头（带容错机制）                         │
│  3. 主循环：                                         │
│     ┌──────────────────────────────────┐            │
│     │ 读取视频帧                        │            │
│     │ ↓                                │            │
│     │ 每 N 帧执行一次检测                │            │
│     │ ↓                                │            │
│     │ 小尺寸图像推理 (320x240)          │            │
│     │ ↓                                │            │
│     │ 坐标还原到原图尺寸                 │            │
│     │ ↓                                │            │
│     │ 指数平滑滤波 (减少抖动)            │            │
│     │ ↓                                │            │
│     │ 绘制绿框 + FPS 显示               │            │
│     │ ↓                                │            │
│     │ 等待键盘事件                      │            │
│     └──────────────────────────────────┘            │
│  4. 资源释放                                          │
└─────────────────────────────────────────────────────┘
                          ↓ 调用
┌─────────────────────────────────────────────────────┐
│              yunet_detector.py (检测器封装)           │
├─────────────────────────────────────────────────────┤
│  • ensure_yunet_model(): 自动下载模型                │
│  • YuNetDetector 类:                                │
│      - __init__: 初始化 OpenCV FaceDetectorYN       │
│      - set_input_size: 设置输入尺寸                  │
│      - infer: 执行推理，返回人脸框坐标                │
└─────────────────────────────────────────────────────┘
```

### 核心技术栈

- **语言**: Python 3.10+
- **视觉库**: OpenCV >= 4.8（使用内置的 YuNet 检测器）
- **模型**: face_detection_yunet_2023mar.onnx
- **系统要求**: Windows / macOS / Linux，带摄像头

### 关键实现细节

#### 1. 管道式处理流程

```
视频捕获 → 图像预处理 → 模型推理 → 坐标还原 → 平滑滤波 → 结果绘制 → 窗口显示
```

每一帧的处理步骤：

1. **读取帧**：从摄像头获取 RGB 图像
2. **降采样**：缩小到 `infer_width x infer_height`（默认 320x240）加速推理
3. **检测**：YuNet 输出人脸框 `[x, y, w, h, confidence, ...]`
4. **坐标还原**：将小图坐标映射回原图尺寸
5. **平滑**：使用指数移动平均（EMA）减少框的抖动
   ```python
   smooth_box = (1 - alpha) * prev_box + alpha * current_box
   # alpha = 0.35，越小越平滑，越大越跟手
   ```
6. **绘制**：在原图上画绿色矩形框 + 文字信息

#### 2. 性能优化策略

- **小图推理**：在 320x240 的缩略图上检测，速度提升 5-10 倍
- **跳帧检测**：默认每 2 帧检测一次，可通过 `--detect_every` 调节
- **智能复用**：只检测最佳人脸框（置信度最高），减少计算量
- **EMA 平滑**：避免逐帧更新导致的视觉抖动

#### 3. 健壮性设计

- **摄像头容错**：尝试多个索引（0/1/2/3），并验证是否能读到真实画面
- **连续失败保护**：连续 5 次读取失败则退出循环，防止死锁
- **资源管理**：`try-finally` 确保摄像头释放和窗口销毁
- **数值安全**：所有除法运算都有非零校验，避免除零错误
- **边界检查**：推理尺寸限制在合理范围内（最小 160x120）

#### 4. 数据流

```
原始帧 (640x480)
    ↓ resize
推理图 (320x240)
    ↓ YuNet 检测
faces[n][15]  # n 个人脸，每个包含 15 个特征
    ↓ 选取最佳
best_face[15]  # [x, y, w, h, confidence, landmarks...]
    ↓ 坐标还原
原图坐标 (x*scale_x, y*scale_y, w*scale_x, h*scale_y)
    ↓ EMA 滤波
smooth_box -> 绘制到屏幕
```

---

## 常见问题

### 摄像头打不开

- 先关闭占用摄像头的软件（微信/QQ/浏览器会议等）
- Windows 设置 → 隐私与安全性 → 摄像头 → 允许桌面应用访问摄像头
- 尝试手动指定摄像头编号：`python .\main.py --camera 0` 或 `--camera 1`

### 安装 mediapipe 失败

本项目实际使用的是 **OpenCV 内置的 YuNet 检测器**，不需要 mediapipe。如果安装失败可以忽略该依赖，或者直接从 `requirements.txt` 中移除。

只需确保安装了：

```bash
pip install opencv-python>=4.8
```

### 想要更轻量

可以在运行时降低分辨率或减少检测频率：

```bash
# 方法 1: 降低分辨率（在 main.py 中修改）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# 方法 2: 运行时参数调节
python .\main.py --detect_every 3 --infer_width 240
```

### 检测框抖动

调整平滑系数（在 `main.py` 中修改 `alpha` 变量）：

- 当前值：`alpha = 0.35`
- 更平滑：改为 `0.2`（但有延迟感）
- 更跟手：改为 `0.5`（但可能抖动）

### 模型下载失败

`yunet_detector.py` 会自动从 GitHub 下载模型文件。如果下载失败：

1. 检查网络连接
2. 关闭代理服务器重试
3. 手动下载模型并放到 `models/face_detection_yunet_2023mar.onnx`

模型地址：

```
https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

---

## 项目结构

```
face-tracking/
├── main.py                 # 主入口，负责流程控制和 UI 渲染
├── yunet_detector.py       # 检测器封装，处理模型加载和推理
├── download_models.ps1     # PowerShell 脚本，用于手动下载模型
├── requirements.txt        # Python 依赖列表
├── README.md              # 项目文档
└── models/                # 模型文件目录（自动生成）
    └── face_detection_yunet_2023mar.onnx
```

---

## 扩展开发建议

如果你想在此基础上扩展功能，可以参考以下方向：

1. **多人脸检测**：当前只取最佳人脸，可改为遍历所有 `faces` 数组
2. **保存截图**：在检测到人脸时调用 `cv2.imwrite()` 保存
3. **口罩检测**：结合其他模型判断是否佩戴口罩
4. **GUI 界面**：使用 tkinter 或 PyQt 制作图形界面，替代命令行参数
5. **日志记录**：添加 logging 模块记录检测历史
6. **API 服务**：用 Flask/FastAPI 封装成 HTTP 接口
