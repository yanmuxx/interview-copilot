# 面试 Copilot

带界面的本机实时面试辅助：导入简历 → AI 自动生成可编辑的"候选人画像" → 开始面试后，
抓取系统播放的面试音频 → Silero VAD 切分 → faster-whisper 本地转写 → LLM 流式输出答题要点。

全程本地转写（不依赖云端 ASR），LLM 走任意 OpenAI 兼容 API。

## 运行（推荐）

```bash
cd interview-copilot
python -m venv .venv        # 首次
.venv\Scripts\pip install -r requirements.txt   # 首次（精确版本）
# 要求与已验证环境 100% 一致时，改用：pip install -r requirements.lock
.venv\Scripts\python app.py
```

`app.py` 会打开一个桌面窗口（WebView2）；若环境不支持则自动退回浏览器打开。

## 界面使用

1. **LLM 设置**：填 API Key（DeepSeek / GLM / Kimi 等 OpenAI 兼容接口），保存；
2. **简历 → 画像**：上传 PDF / DOCX / TXT / MD 简历（可附 JD），点「解析简历」，
   AI 生成候选人画像，**可手动编辑**后保存；
3. 右上角选「简明要点（快）」或「完整回答（慢）」，点「开始面试」；
4. 播放任意面试音频（会议软件 / 视频），问题和答案实时出现在右侧，
   会话记录同时存到 `logs/`。

## 命令行模式（可选）

不用界面时也可直接跑终端版：

```bash
.venv\Scripts\python main.py
```

首次运行会自动下载两个小模型：Silero VAD（2MB）+ whisper base（~145MB，走国内镜像）。

## 调优

| 症状 | 改法 |
|---|---|
| 转写有同音字错（线程→现成） | 往 `config.toml` 的 `asr.hotwords` 里加领域词 |
| 转写整体不准 | 有 NVIDIA 显卡就保持 `model = "large-v3-turbo"`；无 GPU 用 `base`/`small` |
| 问题被切碎 | 调大 `vad.min_silence_ms`（如 600，但出结果会变慢） |
| 咳嗽、短音节被当成问题 | 调大 `vad.min_speech_ms` |
| 出答案慢 | 用「简明要点」模式；或换非思考模型（deepseek-chat） |

**转写实测**（同一段 10.7s 测试语音，RTX 4060 Laptop）：

| 方案 | 转写耗时 | 准确率(45字) |
|---|---|---|
| base · CPU int8 | 0.59s | 39 |
| small · CPU int8 | 2.42s | 37 |
| **large-v3-turbo · GPU float16 + 热词** | **0.4s 内** | **45（全对）** |

GPU 首次加载约需下载 1.6GB 模型（走国内镜像）；`device = "auto"` 会在 GPU 不可用时自动回退 CPU。
换面试方向时记得同步改 `hotwords`（前端岗加"浏览器 渲染 Webpack"，算法岗加"大模型 推理 训练"……）。

延迟预算：面试官说完 → 首条要点出现约 2~5 秒（GPU 转写 <1s + LLM），之后流式滚出。

## 结构

```
app.py             桌面启动器（本地服务 + WebView 窗口，失败退回浏览器）
server.py          FastAPI：配置/简历/画像/启停接口 + SSE 实时事件流
web/index.html     界面（原生 HTML/JS，无构建步骤）
main.py            终端版入口
copilot/capture.py WASAPI 回环采集（系统音频 → 16kHz 单声道）
copilot/vad.py     Silero VAD（ONNX 2MB）+ 语音段切分
copilot/asr.py     faster-whisper 本地转写（int8 量化，CPU）
copilot/advisor.py LLM 答题要点/完整回答（流式，两种模式）
copilot/pipeline.py 可启停的面试流水线（采集/转写/回答各一线程，互不阻塞）
copilot/resume.py  简历文本抽取（PDF/DOCX/TXT/MD）+ LLM 画像提炼
config.toml        模型 / VAD / LLM 配置
persona.md         候选人画像（界面保存后写在这里）
```

## 提醒

真面试中使用属于灰色地带且部分公司明令禁止，建议主要用于**模拟面试陪练与复盘**。

## 路线图（后续可选）

- 置顶半透明悬浮小窗（迷你模式），叠加在会议软件上方
- 全局热键：显隐窗口 / 清屏 / 暂停
- 双声道区分"面试官 / 我"，自动只分析提问
