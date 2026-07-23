---
title: YRobot
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
short_description: Low-latency MiniCPM-o 4.5 conversation for Reachy Mini Wireless
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# YRobot

YRobot 是运行在 Reachy Mini Wireless CM4 上的 MiniCPM-o 4.5 实时视听客户端。它直连
MiniCPM-o-Demo Realtime Gateway，默认地址为：

```text
wss://10.0.16.184:8006/v1/realtime?mode=video
```

这次重构不兼容旧 `/backend` 协议。运行时只保留五个边界：Realtime transport、音频与插话、
视频 latest-frame、单一动作控制器和应用生命周期。

## 体验改进

- 严格等待 `session.queue_done` 后初始化，连续发送 16 kHz mono float32 的一秒真实麦克风
  unit；摄像头只保留最新 JPEG，慢视频不会阻塞音频。
- 24 kHz 模型音频用跨 delta 保持状态的重采样器转换到 Reachy 的 16 kHz 播放链路；播放
  以绝对样本时钟逐个推送 20 ms 帧，初始 lead 为 40 ms，调用耗时不会逐帧累积成卡顿；
  队列始终有界，溢出后截断当前回答而不拼接不连续的句尾。
- 建立首个 WebSocket 前从 Gateway 获取并缓存 `/api/default_ref_audio`，会话显式发送
  `use_tts=true`、`generate_audio=true` 和规范的 `voice.ref_audio_base64`；避免 comni
  忽略根级别旧别名后只返回文本而没有语音。
- `session.created` 前不保留麦克风音频，也不显示倾听姿态；后端真正就绪后才打开一个干净的
  capture epoch，避免初始化期间说的话被截断后延迟送达。
- WebRTC VAD 以 20 ms 检测 XVF AEC 后的人声。用户插话时立即递增 playback epoch、清空
  应用队列并调用 `media.audio.clear_player()`；任何晚到旧音频都会失效，绝不恢复旧尾音。
- 硬件 AEC 之后再用最近实际播放的 PCM 做任意 sample lag 的保守相关性校验；高度匹配的
  残余自声不但不触发本地 VAD，也会从模型上行中置零。与当前播放不匹配的近端用户语音仍按
  2–3 个 VAD frame 触发插话。
- 服务端公开协议没有独立 cancel。插话确认时，YRobot 会立即把已经采集的近端开头补零成
  合法的一秒 control unit 并携带 `force_listen=true`，避免再等最多一秒才通知服务端；
  后续用户语音仍按正常一秒 unit 连续发送。
- Reachy 官方 local ALSA/GStreamer 路径和 XVF3800 硬件 AEC 始终保持双讲；启动后回验官方
  conversation app 的降噪、回声和增益参数，不在机器人说话时关闭麦克风。
- DoA 由独立 10 Hz sensor worker 读取并从头部坐标转换到世界坐标；唯一 50 Hz motion worker
  合成最后说话者朝向、倾听和呼吸姿态，所有轴均限速/限加速度。说话细节由 SDK
  playback-synchronised wobbling 驱动。
- 视频会话在 285 秒请求滚动，优先等待服务端 listening 且本地收播都空闲的轮次边界，并在
  294 秒前强制关闭；重连、滚动和停机都先本地 invalidate-and-flush，再停止 sender，最后
  发送 `session.close`，并等到 `session.closed` 后 Gateway 真正关闭传输、释放 Worker；
  关闭后不会再上传旧输入。

## 安装到 Wireless CM4

先确保 Reachy daemon、Python SDK 和机器人固件版本彼此匹配，并且应用使用官方本地媒体链路。
当前依赖精确固定为审计过的 `reachy-mini==1.9.0`；若机器人 daemon 不是 1.9.0，应同步修改
两者版本，不能只单独升级应用 SDK。

```bash
git clone <your-yrobot-repository>
cd YRobot
./scripts/setup_cm4.sh
```

脚本会创建 Python 3.12 虚拟环境、安装依赖并从 `.env.example` 创建 `.env`。

可先确认 Wireless 音频别名存在：

```bash
aplay -L | grep reachymini_audio_sink
arecord -L | grep reachymini_audio_src
```

不要改用任意 `sounddevice` 设备；这样可能绕过 XVF3800 的扬声器 reference 和硬件 AEC。

## 配置

主要配置都在 `.env`：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `YROBOT_REALTIME_URL` | `wss://10.0.16.184:8006/v1/realtime?mode=video` | Gateway 地址 |
| `YROBOT_SEND_VIDEO` | `1` | 每个音频 unit 附一张 latest JPEG |
| `YROBOT_ENABLE_TTS` | `1` | 获取 Gateway 默认音色并请求语音输出 |
| `YROBOT_TLS_VERIFY` | `0` | 当前 Gateway 使用自签名证书；换可信证书后设为 `1` |
| `YROBOT_FORCE_LISTEN_COUNT` | `1` | 会话启动时强制倾听的 unit 数 |
| `YROBOT_LENGTH_PENALTY` | `1.1` | Duplex 生成参数 |
| `YROBOT_SESSION_ROLLOVER` | `285` | 在 300 秒上限前滚动会话 |
| `YROBOT_RECONNECT_INITIAL` | `0.5` | 初始重连退避秒数 |
| `YROBOT_RECONNECT_MAX` | `8` | 最大重连退避秒数 |

裸 `host:port` 默认按官方部署规范化为 WSS video Realtime URL；明文部署必须显式写
`http://` 或 `ws://`。旧 `/backend` 会被明确拒绝。当前 `10.0.16.184:8006` 的证书是自签名
证书且没有 Subject Alternative Name，因此默认关闭校验；离开可信局域网前应重签带
`subjectAltName=IP:10.0.16.184`（或匹配 DNS SAN）的可信证书，再设置
`YROBOT_TLS_VERIFY=1`。

## 启动

Reachy daemon 已运行时，在 CM4 上执行：

```bash
source .venv/bin/activate
yrobot
```

Reachy dashboard 会通过 `python -m yrobot.main` 进入 `ReachyMiniApp.wrapped_run()`，由官方
stop event 和 local media 生命周期托管；命令行入口仍独立保留为 `yrobot`。CLI 覆盖示例：

```bash
yrobot --no-video
yrobot --url wss://10.0.16.184:8006/v1/realtime?mode=video
yrobot --force-listen-count 1
```

CLI 显式使用：

```python
ReachyMini(
    connection_mode="localhost_only",
    media_backend="local",
    automatic_body_yaw=True,
)
```

## 不可变服务端边界

当前部署已经实测为三层服务：

```text
WSS Gateway :8006 -> Python worker :22400 -> comni C++ backend :22500
```

`http://10.0.16.184:22500/health` 返回 `{"engine":"comni","status":"ok"}` 只说明 C++
backend 存活；客户端必须连接 `wss://10.0.16.184:8006/v1/realtime?mode=video`。C++ runtime
按 backend protocol 直接接收 `payload.config.force_listen_count`。

2026-07-23 最近一次对当前 `.184` 的真实 probe：queue 0.9 ms，
`session.init → session.created` 14987.5 ms，强制 listen 的 input-to-listen 72.9 ms，服务端
`wall_clock_ms` 仅 1.159 ms。此前同一服务的测量为 queue 0.1–0.6 ms、listen
45.0–120.8 ms、init 9.20–15.77 秒；`.187` 上两次真实问题音频的首个 24 kHz 回答在 ready 后
1.58 秒和 2.53 秒到达，均返回完整文本与语音。源码定位表明 comni 每次会话都会重新读取约
5 GB GGUF embedding 数据。服务端被视为完全不可变，因此：

- 约 9–16 秒首次连接及每次 300 秒 rollover 的重新初始化窗口是固定边界，客户端不能消除；
- 单 worker 无法并行预热下一会话，YRobot 只会在空闲轮次优雅滚动并保持明确的 idle 姿态；
- 当前服务的自然语言 system prompt 会破坏 duplex 模板，客户端强制发送空字符串并拒绝
  非空 `YROBOT_SYSTEM_PROMPT`，始终使用服务端内置 persona；
- 客户端默认发送已经在线上 comni 验证过的 `force_listen_count=1`，不修改公开协议的一秒
  输入 cadence；
- Gateway 在 `session.closed` 之后才完成 Worker 回收；客户端等待传输关闭再重连，并以带
  抖动的退避重试偶发 `session_failed`/HTTP 403，无需服务端补丁或重启；
- 所有性能优化仅发生在 Reachy 客户端：采集、打断、本地静音、播放、视频和动作链路。

## 自动验证

```bash
source .venv/bin/activate
python -m pytest
ruff check .
python scripts/probe_realtime.py
reachy-mini-app-assistant check .
```

服务分层可分别检查：

```bash
curl -k https://10.0.16.184:8006/health  # Gateway
curl http://10.0.16.184:22400/health     # worker（仅诊断）
curl http://10.0.16.184:22500/health     # comni backend（仅诊断）
```

Realtime probe 会严格验证 queue 生命周期，发送 16,000 个静音 float32 样本并请求
`force_listen`，随后等待 `kind=listen`，打印服务端 `wall_clock_ms`。它不会验证扬声器、AEC、
DoA 或机械动作，这些必须在真实 Wireless 机器人上验收。

## 实机验收

建议固定音量、距离和房间，至少测试以下场景：

1. 机器人独自播音：VAD 不应被自己的声音触发。
2. 用户独自说话：输入持续上传，首响没有额外客户端预滚。
3. 用户在机器人说话时插话：可闻旧音频应在 100 ms 内停止，之后一次也不能恢复。
4. 用户在 `listen` 确认后仍继续说：机器人保持退让，直到用户安静。
5. 机器人转头、天线运动和环境噪声：不能误触发插话，motion tick 不受 DoA USB 抖动影响。
6. 运行 30 分钟：285 秒滚动无跨会话旧音频，所有 queue 保持有界。

退出时会打印媒体指标，包括 VAD 插话次数、自声抑制 frame、clear latency、audio delta 到
speaker latency、force dispatch、interrupt 到 listen、队列丢弃和服务端 `wall_clock_ms`。若
`clear_player()` 失败，YRobot 会 fail-safe 停止会话，因为此时已无法保证旧音频不会继续播放。

详细设计、延迟预算和状态机见 [plan.md](plan.md)。

## 主要资料

- [MiniCPM-o Realtime API overview](https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/)
- [MiniCPM-o video full-duplex](https://minicpmo45.modelbest.cn/docs/en/realtime-api/video/)
- [MiniCPM-o Realtime examples](https://minicpmo45.modelbest.cn/docs/zh/realtime-api/examples/)
- [Reachy Mini AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md)
- [Reachy Mini Python SDK](https://huggingface.co/docs/reachy_mini/SDK/python-sdk)
- [Reachy Mini API](https://huggingface.co/docs/reachy_mini/API/reachymini)
- [Reachy Mini conversation app](https://huggingface.co/blog/local-reachy-mini-conversation)
