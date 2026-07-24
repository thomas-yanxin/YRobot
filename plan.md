# YRobot v3 — 全双工 MiniCPM-o 4.5 × Reachy Mini Wireless

从零重写。目标：把官方 Realtime API（https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/）
的全双工能力完整发挥到 Reachy Mini Wireless 上，四个痛点各有明确对策。

## 1. 四个痛点 → 对策

| 痛点 | 根因 | 对策 |
|---|---|---|
| 端到端延迟高 | 1 s 上行分块 + 播放缓冲过大 | 500 ms 分块（实测回复延迟从 +1.7 s 降到 +1.1~1.3 s；250 ms 会破坏 turn-taking）；`mode=audio`（600 s 上限，仍接受视频帧）；自适应 0.25–0.8 s preroll，设备积压 ≤1.5 s |
| 打断不及时 / 旧音频复播 | 服务端 `force_listen` 不可靠：生成不会真正停止，且安静后常"续播"旧独白 | 打断完全由客户端兜底：VAD onset → 立即 `clear_player()` + 整轮丢弃 + 每个分块携带 `force_listen`；只有连续 2 个"干净 listen"（用户安静 ≥0.7 s 且不在我方 force 的 1.2 s 内）才解除丢弃，12 s 封顶 |
| 被自己的回声/动作声误打断 | AEC 残余回声在一个音节内跟上远端包络，静态阈值必被大声 TTS 戳穿（2026-07-22/24 实机日志） | 两级打断：① EchoGuard 用「已播音频包络 + 学习到的泄漏比 + 3 dB」预测残余，候选必须超过预测；② 候选只 duck（清设备队列、暂存未播尾音），0.6 s settle（管线+采集延迟实测 ≥286 ms）后扬声器已静、回声路径已死，人声持续 60 ms 才真正销毁本轮，否则无损恢复播放并把泄漏比 +1 dB（同级残余不再触发）。电机噪声另有非对称噪声底 + 60 ms 连续确认兜底 |
| 动作不拟人 | 动作源互相打架、无统一节奏 | 单一 50 Hz 动作 owner（呼吸/扫视/姿态全部临界阻尼合成）；说话嘴动用 SDK 官方 `enable_wobbling()`（daemon 侧与扬声器 PTS 同步）；body yaw 交给 `set_automatic_body_yaw(True)` 跟随头部 |
| DoA 不灵敏 | 用固件 speech 标志做门控（它在 AEC 之前，机器人自己说话也触发） | 12 Hz 独立线程轮询 `DOA_VALUE_RADIANS`，只在**本地 VAD 判定用户在说话**时采样，1 s 窗口圆均值 + 死区，头相对角换算成世界 yaw 后交给动作 owner 平滑转过去；可叠加 daemon 人脸跟踪细修 |

## 2. 协议要点（来自官方文档 + 实测）

- `wss://…/v1/realtime?mode=audio`：上行 base64 float32 16 kHz mono，下行 24 kHz；
  `session.queue_done` → `session.init` → `session.created`（~14 s，服务端固定成本）。
- delta `kind ∈ {listen, text, audio}`；**只有 listen 是语义轮边界**；text/audio 不一一对应。
- `mode=audio` 仍接受 `video_frames`（base64 JPEG），且会话上限 600 s（video 只有 300 s）。
- system_prompt 首行必须是训练句 `You are a helpful assistant.`，第二行放简短人设
  （自由人设会让 Qwen3 底座漂出双工分布、`<think>` 泄漏）。
- kv 预算 ~8192：视觉 64 tok/帧 是大头 → 机器人独自说话时不发帧；活跃 1 fps、空闲 0.2 fps；
  时间/kv 双预算到点后，只在安静的 listen 边界轮换会话。

## 3. 模块（6 文件，单一职责）

```
config.py    环境变量 → 冻结配置（URL 归一化、所有可调参数）
realtime.py  网关协议客户端（排队/init/收发/关闭）+ ThinkFilter
turn.py      打断状态机（纯逻辑、单测覆盖）
audio.py     VoiceDetector(VAD+噪声底) / MicChunker / LinearResampler / Speaker(epoch 播放)
motion.py    SoundCompass(DoA) + Choreographer(50 Hz 唯一动作 owner)
main.py      ReachyMiniApp 接线、会话生命周期、CLI
```

线程：mic 上行（主循环）、ws 收、扬声器、动作 50 Hz、DoA 12 Hz。跨线程只传不可变数据。

## 4. 验收（实机）

1. 打断 30 次：物理静音 ≤100 ms，旧音频零复播。
2. 机器人独白 + 头/身体大幅运动 30 次：零自我打断。
3. 侧后方说话：1 s 内头转向说话人，无阶跃。
4. 连续 30 min，≥3 次会话轮换，轮换期间保持 idle 姿态。
