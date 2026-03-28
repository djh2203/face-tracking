# 极简人脸框跟踪（零基础版）

你只需要照着复制命令执行，就能打开摄像头并在画面里画出人脸框。

## 你将得到什么

- **轻量级**：不训练模型，不写复杂代码
- **实时**：摄像头画面 + 人脸框 + FPS
- **一键退出**：按 `Q` 或 `ESC`

## 1) 安装 Python（只做一次）

1. 打开微软商店或去官网安装 **Python 3.10+**
2. 安装时勾选 **Add Python to PATH**

安装好后，打开 PowerShell 输入：

```bash
python --version
```

能看到版本号就 OK。

## 2) 安装依赖（第一次运行前做一次）

在 PowerShell 里进入这个项目目录：

```bash
cd c:\github\face-tracking
```

建议先创建虚拟环境（更干净，不会把系统 Python 搞乱）：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

## 3) 运行

```bash
python .\main.py
```

看到窗口后，把脸放到摄像头前，会出现绿色框。

## 常见问题

### 摄像头打不开

- 先关闭占用摄像头的软件（微信/QQ/浏览器会议等）
- Windows 设置 → 隐私与安全性 → 摄像头 → 允许桌面应用访问摄像头

### 安装 mediapipe 失败

先确认 Python 版本是 3.10/3.11/3.12 这类常见版本，然后重试：

```bash
pip install -r requirements.txt
```

### 想更快（更轻量）

你可以在 `main.py` 里把分辨率从 `640x480` 再降一点（例如 `480x360`），会更快。
