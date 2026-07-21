# YRobot 重构计划：Reachy Mini × MiniCPM-o 4.5 Omni

状态：已确认，正在实施。

## 1. 目标理解

把 YRobot 从零重构为一个小而可靠的 Reachy Mini Python 应用：

- 直连已经部署好的 `llama-omni-server`：
  `wss://10.0.16.187:28099/backend`。
- 以 `full_duplex` 模式持续上传 Reachy 的麦克风和摄像头数据。
- 实时播放 MiniCPM-o 4.5 返回的语音；模型负责听、说切换和打断语义。
- 对话时让机器人有自然动作：朝向说话人、说话时随语音轻微摆动、在监听/空闲时保持
  克制的生命感。
- 优先复用 Reachy Mini SDK 与 MiniCPM 官方协议，不复制 conversation app 的大型框架。
- 旧实现没有兼容要求；未确认前不删除现有代码。

2026-07-21 已做只读健康检查，服务返回：
`{"engine":"comni","status":"ok"}`。

## 2. 已核实的事实

### llama.cpp-omni

- 数据通道是 `/backend`；首条消息必须是 `session.init`。
- `full_duplex` 每个时间片发送一条 `input.append`：1 秒左右的 16 kHz、单声道、
  float32 裸 PCM，可附一张 base64 JPEG。
- 服务返回 `response.output.delta`，`kind` 为 `listen`、`text` 或 `audio`。
- C++ Token2Wav 实际输出 24 kHz float32 PCM，但 WebSocket 事件不携带采样率；客户端需
  将 24 kHz 下采样到 Reachy 播放设备的 16 kHz。
- 服务当前只允许一个活动会话；WebSocket 断开时后端会清理会话并保留已加载模型。

### Reachy Mini SDK

- `mini.media.get_audio_sample()` 给出 16 kHz、双声道 float32；上传前只需混为单声道。
- `mini.media.get_frame_jpeg()` 可直接复用，避免引入 OpenCV/Pillow 编码流水线。
- `mini.media.push_audio_sample()` 原生播放 16 kHz float32。
- `mini.enable_wobbling()` 已提供与实际播放音频同步的说话动作，不应再自建一套波形动画。
- `mini.media.get_DoA()`、`look_at_world()` 是官方的说话人朝向路径。
- 平滑动作使用 `goto_target()`；不需要实时姿态融合时，不引入 30/100 Hz `set_target()`
  控制环。

参考：

- [Reachy Mini AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md)
- [MiniCPM-o 4.5 llama.cpp-omni 文档](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/minicpmo_4_5_llamacpp_omni_zh.md)
- [MiniCPM-o 官方 Realtime 示例](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/main/examples/realtime)
- [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app)
- [Reachy Mini Python SDK](https://huggingface.co/docs/reachy_mini/SDK/python-sdk)
- [Reachy Mini DoA 示例](https://github.com/pollen-robotics/reachy_mini/blob/main/examples/sound_doa.py)

## 3. 最小架构

```text
Reachy mic (16k stereo)
  -> mono + 1 秒分片
  -> Omni WebSocket /backend ----------------------+
                                                     |
Reachy camera -> 当前 JPEG，每个时间片最多一张 ------+-> MiniCPM-o 4.5
                                                     |
Reachy speaker <- 24k -> 16k 重采样 <- audio delta --+

DoA -> 单一串行动作 worker -> look_at_world/goto_target
播放音频 -> Reachy SDK enable_wobbling()
```

计划只保留四个职责明确的 Python 模块：

```text
yrobot/
  config.py   # 少量环境配置与校验
  omni.py     # 协议编解码、WS 会话、重连
  robot.py    # Reachy 音视频适配、播放、DoA/动作
  main.py     # ReachyMiniApp 与独立 CLI 生命周期
```

测试只覆盖稳定边界：协议消息、PCM 编解码与重采样、URL/TLS 配置、DoA 坐标转换、断线重连
和假机器人上的端到端数据流。

## 4. 明确删除的复杂度

- 删除自建 Event Bus、复杂状态机和多层 pipeline。
- 删除 Web UI、FastAPI、静态前端；第一版以日志和 Reachy dashboard 生命周期为准。
- 删除本地 ASR、VAD、LLM、TTS、AGC、情绪关键词分类和自定义 30/100 Hz 动作合成器。
- 删除 gateway (`:8006`) 兼容；当前目标明确是直连 raw backend (`:28099`)。
- 删除几十个实验性环境变量，只保留 endpoint、TLS、prompt、视频开关/尺寸、日志等级。
- 不复制 conversation app 的 tool registry、profile、MCP、Gradio 和 motion manager。

## 5. 复用策略

- 音视频设备、AEC、DoA、动作安全限制、语音摆动：复用 `reachy-mini` SDK。
- WebSocket 消息格式与 1 秒发送节奏：按 MiniCPM-o-Demo 官方 probe 实现。
- 输出下采样：使用 Reachy SDK 已依赖的 `scipy.signal.resample_poly`。
- 可选情绪/舞蹈若进入后续范围，复用
  `pollen-robotics/reachy-mini-emotions-library` / dances library，不复制动作资产。

## 6. 实施顺序

1. 根据下面的回答写入 `agents.local.md`，记录硬件与运行位置。
2. 用 `reachy-mini-app-assistant check` 校验现有应用骨架；保留必要的 HF/Reachy 元数据。
3. 新建最小 `yrobot` 包及单元测试，再替换入口和依赖。
4. 删除旧包、旧前端和过时测试，保证仓库只剩必需内容。
5. 运行 `pytest`、`ruff` 和 `reachy-mini-app-assistant check`。
6. 用 fake Reachy/fake Omni 做自动化 smoke test。
7. 最后连接真实 `wss://10.0.16.187:28099/backend` 做协议 smoke test；物理音频、DoA 和动作
   由你在真实机器人旁确认，测试时先使用低音量和小幅动作。

## 7. 已确认的选择

### Q1：硬件和运行位置

Reachy Mini 是哪一版，YRobot 准备运行在哪里？

答案：`Wireless 的 CM4`

建议：如果是 Wireless 且与 `10.0.16.187` 同一局域网，优先运行在 CM4；机器人只做 I/O，
模型推理在远端服务器。

### Q2：第一版动作范围

答案：`A；后续逐步扩展到 B`

- A（建议）：自然自动动作——DoA 转向、SDK 语音摆动、轻微监听/空闲姿态。
- B：再加入模型主动选择的 `dance` / `play_emotion` / `move_head` 等命名动作。

说明：raw `llama-omni-server` 的 full-duplex 协议只输出 listen/text/audio，没有原生 tool
call。选择 B 需要额外设计一个动作控制侧通道或结构化标记协议，会增加复杂度，也需要确认
你的服务端是否愿意配合修改。

## 8. 验收标准

- 连续音频和约 1 fps 视频可稳定上传，网络慢时不会无限堆积。
- 语音 delta 连续播放，无明显变速、爆音或无限缓冲。
- 用户可在机器人说话时继续讲话，模型仍持续收到麦克风时间片。
- 用户说话连续触发 ReSpeaker VAD 时，立即清空应用与 SDK 播放队列；下一段输入携带一次
  `force_listen`，让 MiniCPM-o 停止当前生成并继续接收用户语音。
- 有声音朝向和说话摆动，但动作调用串行、幅度受 SDK 安全限制。
- 断网后有界重连；Ctrl-C/应用停止能释放媒体、关闭 WS、停止动作并安全回中性姿态。
- 核心仓库结构明显小于当前版本，测试和文档与实际协议一致。
