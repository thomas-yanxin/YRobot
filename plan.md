# YRobot 重构计划：Reachy Mini × MiniCPM-o 4.5 Omni

状态：拟人动作优化已实施；待 Wireless 实机观察后微调幅度。

## 1. 目标理解

把 YRobot 从零重构为一个小而可靠的 Reachy Mini Python 应用：

- 直连已经部署好的 `llama-omni-server`：
  `wss://10.0.16.184:28099/backend`。
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

- `mini.media.get_audio_sample()` 给出 XVF3800 处理后的 16 kHz、双声道 float32；播放期间也
  持续读取真实 PCM，上传前只需混为单声道，不能用全零替代。
- `mini.media.get_frame_jpeg()` 可直接复用，避免引入 OpenCV/Pillow 编码流水线。
- `mini.media.push_audio_sample()` 原生播放 16 kHz float32。
- `mini.enable_wobbling()` 已提供与实际播放音频同步的说话动作，不应再自建一套波形动画。
- `mini.media.get_DoA()`、`look_at_world()` 是官方的说话人朝向路径。
- YRobot 使用 DoA 做朝向和连续的轻量姿态叠加；DoA speech 是 PRE-AEC 信号，机器人播音时
  恒为高，因此不参与 double-talk 判定——打断由 post-AEC 电平候选加静音验证独立确认。
  按 SDK 的实时控制原则使用
  唯一的 `set_target()` 控制点，当前为 50 Hz，相位对齐并按实际 `dt` 限速。
- SDK 的 `enable_wobbling()` 继续负责按真实音频播放时间戳生成说话动作；应用层只叠加更小的
  呼吸、倾听点头、短暂移视和天线动作，避免重复驱动同一类说话动画。

参考：

- [Reachy Mini AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md)
- [MiniCPM-o 4.5 llama.cpp-omni 文档](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/minicpmo_4_5_llamacpp_omni_zh.md)
- [MiniCPM-o 官方 Realtime 示例](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/main/examples/realtime)
- [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app)
- [Reachy Mini Python SDK](https://huggingface.co/docs/reachy_mini/SDK/python-sdk)
- [Reachy Mini DoA 示例](https://github.com/pollen-robotics/reachy_mini/blob/main/examples/sound_doa.py)

## 3. 最小架构

```text
Reachy speaker -> XVF3800 far-end reference
                         |
Reachy mic (16k stereo) -> XVF3800 AEC -> mono + 1 秒真实 PCM 分片
  -> Omni WebSocket /backend ----------------------+
                                                     |
Reachy camera -> 当前 JPEG，每个时间片最多一张 ------+-> MiniCPM-o 4.5
                                                     |
Reachy speaker <- 24k -> 16k 重采样 <- audio delta --+

DoA -> 20 Hz 感知 / 50 Hz 单一串行动作 worker -> look_at_world/set_target
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
- 删除本地 ASR、LLM、TTS、情绪关键词分类和大型动作编排框架；只保留小幅、可中断的实时
  姿态合成层。
- 删除 gateway (`:8006`) 兼容；当前目标明确是直连 raw backend (`:28099`)。
- 删除几十个实验性环境变量，只保留 endpoint、TLS、prompt、视频开关/尺寸、日志等级。
- 不复制 conversation app 的 tool registry、profile、MCP、Gradio 和 motion manager。

## 5. 复用策略

- 音视频设备、AEC、DoA、动作安全限制、语音摆动：复用 `reachy-mini` SDK。
- WebSocket 消息格式与 1 秒发送节奏：按 MiniCPM-o-Demo 官方 probe 实现。
- 输出下采样：使用带跨 delta 滤波状态的流式 FIR 重采样，避免每段独立补零造成块边界噪声。
- 可选情绪/舞蹈若进入后续范围，复用
  `pollen-robotics/reachy-mini-emotions-library` / dances library，不复制动作资产。

## 6. 实施顺序

1. 根据下面的回答写入 `agents.local.md`，记录硬件与运行位置。
2. 用 `reachy-mini-app-assistant check` 校验现有应用骨架；保留必要的 HF/Reachy 元数据。
3. 新建最小 `yrobot` 包及单元测试，再替换入口和依赖。
4. 删除旧包、旧前端和过时测试，保证仓库只剩必需内容。
5. 运行 `pytest`、`ruff` 和 `reachy-mini-app-assistant check`。
6. 用 fake Reachy/fake Omni 做自动化 smoke test。
7. 最后连接真实 `wss://10.0.16.184:28099/backend` 做协议 smoke test；物理音频、DoA 和动作
   由你在真实机器人旁确认，测试时先使用低音量和小幅动作。

## 7. 已确认的选择

### Q1：硬件和运行位置

Reachy Mini 是哪一版，YRobot 准备运行在哪里？

答案：`Wireless 的 CM4`

建议：如果是 Wireless 且与 `10.0.16.184` 同一局域网，优先运行在 CM4；机器人只做 I/O，
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
- 用户可在机器人说话时继续讲话，模型持续收到未经静音替换的 XVF post-AEC 麦克风时间片。
- MiniCPM-o 优先自主输出 `listen/speak`。能量检测只产生**候选**，绝不直接废弃轮次：
  post-AEC 电平须持续 60 ms 超过"预期回声残余"——由刚推送的播放包络加上在线学习的
  残余比值实时预测（慢速学习的残余底噪 + 6 dB 只作为下限；真机验证 2026-07-22：
  残余瞬时跟随远端音节，底噪每秒只爬 9 dB，任何静态余量都会被响亮音节击穿）。
  候选确认后先**静音验证（duck-and-verify）**：清空设备队列但保留未播尾音，扬声器
  静默、回声路径消失后 post-AEC 仍有持续人声才提交破坏性清空并发送 `force_listen`；
  否则原位续播尾音（误触发的代价从整轮 12 秒变成一次约 1 秒的顿挫），同时抬高学习
  到的残余水平避免同电平反复触发。验证沉降期与包络回看窗都必须按共享 GStreamer
  管线的实测延迟标定（启动日志报告 min 286 ms / max 1.26 s；真机验证 2026-07-22：
  150 ms 沉降期采到的还是在途回声、0.45 s 回看窗错过了正在空气里响的响亮帧，
  两者叠加把一次回声误触发"验证通过"了）。DoA speech 是 PRE-AEC 信号，机器人
  播音期间恒为高，永不参与双讲判定。打断提交的瞬间把采集缓冲里未满一秒的**真实用户语音**
  立即入队、带着 `force_listen` 标志发出（真机验证：改发静音控制片会让模型"没听到人"，
  在 listen 确认后直接继续讲自己的故事，且随后可能长时间不回话）。`force_listen` 标志
  持续附加到后续片段直到 `listen` 确认，超时 3 秒自动放弃；被打断轮次是"整轮废弃"——
  服务端一个 turn 会超前 10+ 秒突发送流，其音频一律丢弃到下一个 `listen` 边界（模型
  真正停止说话）为止，绝不中途复活。新会话建立时清除遗留打断状态。**用户没说完之前
  持续退让**（真机验证：listen 确认后模型往往下一个切片就开始新话语，直接压着还在
  说话的用户讲）：listen 确认到达时若 0.7 秒内仍有用户语音信号，保持静音不解除；
  静音期间模型音频再到达就立即重发 `force_listen`（限频 1 次/秒，绕过能量检测器
  arm + 确认 + 验证的重检延迟）；用户安静后的第一个 `listen` 边界恢复播放。
  **listen 边界已过且用户安静超过退让保持时，新到达的模型音频直接解除废弃并播放**
  ——不再等下一个 `listen` 边界（长独白轮次可能一分钟都没有边界，真机验证
  2026-07-22：一次误打断让 49 秒的故事整轮静默）；12 秒上限防止环境噪声导致
  永久静音。安静的普通 `listen` 保留短缓冲句尾，已确认的用户插话则清空 GStreamer；
  `clear_player` 会对共享的录音+播放管线做 PAUSED→flush→PLAYING 循环，只允许播放线程
  （唯一推流者）执行，否则管线可能永久卡死。
- **`response_id` 与 `response.done` 是 1 秒时间片级别的，不是句子级别**：一句话跨越多个
  连续 speak 切片，每片各有一个 `done`；只有 `listen` delta 才是真正的句子边界。因此切片
  `done` 不改会话状态、不解除废弃、不衰减预滚，文本片段聚合到 `listen` 再整句打日志，
  否则句中切片会被误判成新回复起点、插入预滚等待，把一句话拆成多句读。
- 播放预滚只作用于句子起点：真实 TTS 供给缺口把预滚抬高到缺口加余量，每个开始播放的
  句子衰减 20 ms，收敛下限 200 ms（1 秒切片节奏下更薄的缓冲会在片界断供）；句中断供
  立即续播且不重置重采样器，等待只会拉长可闻空洞。发送慢于 250 ms 时下一片只发音频。
- 上行 AGC：MiniCPM-o 会把安静的近端语音当背景音而不回答，上传片段按平滑语音电平估计
  提升到 0.12 rms（增益只增不减，上限 8 倍）；机器人说话期间冻结估计，防止残余回声拉低
  语音基准。双讲检测读取的是 AGC 之前的原始 post-AEC 电平。
- 有声音朝向、连续微动作、倾听反馈和按播放同步的说话摆动；动作调用串行、幅度和速度均有界。
- 断网后有界重连；Ctrl-C/应用停止能释放媒体、关闭 WS、停止动作并安全回中性姿态。
- 核心仓库结构明显小于当前版本，测试和文档与实际协议一致。
