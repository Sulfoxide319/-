# 第一版网页原型推进方案

## 目标

验证“语音活字印刷”核心链路：已有录音被拆成短语块，用户通过网页把短语块拖入输出轨，系统按输出轨顺序拼接并做基础后处理，最终生成可听懂的新音频。

## 本机适配结论

- Windows 本地开发。
- Python 3.10 可用。
- Node 24 可用。
- FFmpeg/FFprobe 已安装。
- RTX 4060 Laptop 8GB 显存可用于 WhisperX，但当前全局 PyTorch 是 CPU 版。
- 当前原型支持 fallback 转写，项目内 `.venv` 已补齐 WhisperX + CUDA 版 PyTorch 后可启用自动转写。

## 架构

```text
apps/web    React + Vite 工作台
apps/api    FastAPI 后端
data        本地音频、转写 JSON、拼接结果
```

## 已实现基础功能

- 上传音频。
- 浏览器录音。
- 手工转写文本生成短语块。
- WhisperX 调用入口。
- 多录音轨展示。
- 短语块拖拽到输出轨。
- 输出轨排序、删除、清空。
- 插入短停顿、长停顿。
- 短语起止点微调。
- 拼接生成 wav/mp3。
- 基础后处理：切片 margin、fade in/out、crossfade、RMS 轻度统一、最终 normalize。
- 预览和下载结果。

## WhisperX 接入

后端入口位于 `apps/api/transcription.py`：

```text
transcribe(audio_path, manual_text, prefer_whisperx)
```

前端启用 `WhisperX` 开关后会传 `autoTranscribe=true`。如果环境中没有 `whisperx`，后端自动退回 fallback，不阻塞 UI 和拼接链路。

推荐 GPU 环境只安装在项目 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r apps/api/requirements-whisperx-cu128.txt
.\.venv\Scripts\python.exe scripts\check_env.py
```

注意：WhisperX 3.8.6 约束 `torch~=2.8.0`，因此本项目固定使用 `torch==2.8.0+cu128`，不要先装最新 torch 再装 WhisperX，否则 pip 会把 torch 替换回 CPU 版。

## 验收清单

- `npm run build` 通过。
- `python -m compileall apps/api` 通过。
- `/api/health` 返回 ok。
- `/api/health` 返回 `whisperxAvailable=true` 和 `cudaAvailable=true`。
- 网页可在 `http://127.0.0.1:3000` 打开。
- 页面无控制台错误。
- 上传音频后出现短语块。
- 输出轨能生成 wav/mp3。

## 下一轮优先级

1. 安装并验证 CUDA 版 PyTorch + WhisperX。
2. 将 WhisperX 中文输出的 word/segment 时间戳真实接入短语块。
3. 增加波形显示和切点吸附。
4. 增加片段单独试听接口，避免依赖浏览器 `#t=` 行为。
5. 增加项目内虚拟环境脚本和一键启动脚本。
