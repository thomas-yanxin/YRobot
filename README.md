# YRobot

面向 Reachy Mini Wireless 的 MiniCPM-o 4.5 实时语音、视觉与动作交互应用。
默认连接：

```text
wss://10.0.16.184:8006/v1/realtime?mode=video
```

## 这版解决了什么

- 16 kHz 麦克风按 500 ms 上行，插话控制仍按 20 ms 检测并立即发出；
- 插话、断线和 session rollover 都会递增 playback epoch，同时清空应用队列和
  Reachy GStreamer player，旧回答无法跨 turn 或 session 恢复；
- 使用 AEC 后的第 0 通道、WebRTC VAD 和近期播放 PCM 相关性共同抑制自回声误打断；
- 麦克风、相机、DoA、扬声器、WebSocket 和动作控制互相隔离，慢相机/USB 调用不会
  阻塞音频热路径；
- 始终使用 `mode=video`，每张新 JPEG 只随一个音频单元发送，避免音频模式忽略图像
  或重复图像消耗上下文；
- WebSocket sender 在 `session.close` 前停止并 await，客户端等待关闭确认；在 video
  模式 300 秒硬上限前主动 rollover，并根据 `kv_cache_length` 提前保护上下文；
- 唯一 50 Hz 动作 owner 使用真实 `dt`、限速和限加速度，DoA 在独立线程采样。

```text
mic/AEC ──20 ms VAD──► bounded 500 ms uplink ──┐
camera ──latest JPEG, adaptive cadence─────────┼──► MiniCPM-o video realtime
DoA ──latest bearing──► 50 Hz motion owner     │
speaker ◄──20 ms bounded playback + epoch──────┘
```

## 安装

在 Reachy Mini Wireless CM4 上：

```bash
./scripts/setup_cm4.sh
source .venv/bin/activate
```

或者手动安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

项目固定使用 `reachy-mini==1.9.0`。应用请求 Reachy 的 `local` media backend，并以
`localhost_only` 方式连接 Wireless daemon。

## 运行

```bash
source .venv/bin/activate
yrobot
```

也可以从 Reachy dashboard 启动注册的 `yrobot` 应用。主要配置见
[.env.example](.env.example)；旧的 `OMNI_*`、`AUDIO_*` 和 `MOTION_*` 变量不会参与
新运行时配置。

关键默认值：

| 配置 | 默认值 | 说明 |
|---|---:|---|
| `YROBOT_CHUNK_MS` | `500` | 500–1000 ms，20 ms 的整数倍 |
| `YROBOT_PLAYBACK_LEAD_S` | `0.120` | 有界首播缓冲 |
| `YROBOT_FRAME_ACTIVE_S` | `1.0` | 用户说话及其后 3 秒的图像间隔 |
| `YROBOT_FRAME_IDLE_S` | `5.0` | 空闲图像间隔 |
| `YROBOT_KV_SOFT` / `YROBOT_KV_HARD` | `6500` / `7800` | 8192 KV 上限前软/硬切换 |
| `YROBOT_SESSION_ROLLOVER` | `280` | 300 秒硬上限前的软切换点 |

`YROBOT_SYSTEM_PROMPT` 必须保持 MiniCPM-o duplex 模板使用的精确句子：
`You are a helpful assistant.`。不要追加长 persona；行为引导应放在用户对话中。

## 验证

无硬件单元与并发回归：

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

Gateway 协议探针：

```bash
python scripts/probe_realtime.py --tls-no-verify
```

带一张 JPEG 检查 video 输入：

```bash
python scripts/minicpmo_video_smoke.py \
  --tls-no-verify \
  --image /path/to/frame.jpg
```

video Realtime API 的单 session 硬上限是 300 秒。YRobot 可以自动、干净地恢复持续
服务，但服务端如果只有一个 worker，创建下一 session 的模型初始化空窗无法仅靠客户端
完全消除；要无缝切换需要 Gateway 提供并行预热或 session resume。
