# 实时人脸框跟踪

## 1) 安装 Python

1. 打开微软商店或去官网安装 **Python 3.10+**
2. 安装时勾选 **Add Python to PATH**

安装好后，打开 PowerShell 输入：

```bash
python --version
```

能看到版本号就 OK。

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

### 更轻量

你可以在 `main.py` 里把分辨率从 `640x480` 再降一点（例如 `480x360`），会更快。
