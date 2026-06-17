# 语音活字印刷拼接原型

本仓库是第一版本地网页原型：上传或录制音频，生成短语块，把短语拖入输出轨，执行基础拼接和后处理后导出新音频。

## 本机推荐启动方式

```powershell
cd C:\Users\Sulfoxide\Documents\拼接测试
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r apps/api/requirements.txt

cd apps\web
npm install
cd ..\..
```

启动后端：

```powershell
.\.venv\Scripts\activate
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

启动前端：

```powershell
cd apps\web
npm run dev
```

打开 http://127.0.0.1:3000。

## WhisperX / CUDA

后端已经预留 WhisperX 接口。如果当前环境没有安装 WhisperX，转写接口会使用手工输入的文本或文件名生成短语块，确保拼接链路可以先跑通。

为了不影响本机全局 Python，CUDA 版 PyTorch 和 WhisperX 只安装到项目内 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r apps/api/requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r apps/api/requirements-whisperx-cu128.txt
.\.venv\Scripts\python.exe scripts\check_env.py
```

当前本机验证结果：`.venv` 内 `torch 2.8.0+cu128`、CUDA runtime `12.8`、WhisperX `3.8.6`，可识别 RTX 4060 Laptop GPU。

启动后端时也要显式使用 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```
