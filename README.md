---
title: YRobot
description: MiniCPM-o 4.5 full-duplex conversation for Reachy Mini Wireless
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# YRobot

YRobot 是为 Reachy Mini Wireless（CM4）重写的实时视听对话应用。它直接使用
MiniCPM-o 4.5 Video Realtime API，不经过级联 ASR/LLM/TTS，也不复用旧版 YRobot
的音频、会话或动作实现。

核心保证：

- 麦克风持续采集，严格按官方格式发送 16 kHz、mono、F32LE、1 秒音频单元；
- 服务器音频一到即进入独立播放线程，24 kHz 重采样到 Reachy 本地媒体的 16 kHz；
- 用户插话先在本地执行 `clear_player()`，再用 epoch 栅栏丢弃旧响应，并持续发送
  `force_listen=true`，直到收到 `kind=listen`；
- 回声参考只记录真正交给扬声器的音频，配合 XVF3800 硬件 AEC、WebRTC VAD 和
  自适应噪声门，避免机器人被自己的声音打断；
- DoA、相机、网络、采集、播放和动作各自独立；只有一个 50 Hz 控制器可以写电机；
- DoA 按官方示例用实时完整头部姿态变换到世界坐标，不用“最后一次指令角”冒充实测姿态；
- 视觉固定为最新 640 px JPEG、最多 1 fps；网络拥塞时只保留最新输入，不补发旧音频。

## 安装与运行

目标环境是 Reachy Mini Wireless 的 CM4，Python 3.11/3.12，Reachy Mini SDK 1.9.0；
建议将机器人固件升级到 2.1.4，以使用当前的音频、DoA 与动作修复。

```bash
bash scripts/setup_cm4.sh
source .venv/bin/activate
cp .env.example .env
yrobot
```

也可由 Reachy Mini dashboard 启动 `YRobot` 应用。应用强制申请 `local` media
backend，以保留 Wireless 的 XVF3800 硬件 AEC 和扬声器远端参考；不要改为独立的
`sounddevice` 输入/输出。

默认连接官方端点：

```text
wss://minicpmo45.modelbest.cn/v1/realtime?mode=video
```

若使用同协议的局域网 Gateway，只需修改：

```dotenv
YROBOT_REALTIME_URL=wss://gateway.local:8006/v1/realtime?mode=video
YROBOT_TLS_VERIFY=0
```

其余可调项见 [.env.example](.env.example)。协议采样率、1 秒上行单元、640 px /
1 fps 视频、50 Hz 动作和 300 秒会话上限是刻意固定的，不提供偏离官方协议或硬件
控制频率的配置。

## 运行结构

```text
XVF3800 mic ─┬─ 20 ms VAD + echo guard ── local barge-in ── clear_player()
             └─ exact 1 s F32LE units ─── MiniCPM-o Realtime
camera ───────── latest JPEG @ 1 fps ────────┘

MiniCPM-o audio ─ epoch fence ─ 24→16 kHz ─ bounded player ─ Reachy speaker
local VAD + DoA ─ smoothed attention target ─ 50 Hz single motion writer
```

MiniCPM-o 会话严格遵循：

```text
session.queue_done → session.init → session.created
input.append* ↔ response.output.delta(listen|text|audio)
session.close → session.closed
```

Video session 在 300 秒处会被服务器关闭。YRobot 在 285 秒主动结束并创建新会话；
Realtime API 没有上下文迁移接口，因此跨会话不会伪造“无缝续接”。

## DoA 的物理边界

Wireless 麦克风阵列只返回水平角，`0=左、π/2=正前/正后、π=右`。线阵本身无法仅靠
DoA 区分正前和正后，也无法估计俯仰角。YRobot 会用本地近端语音门控、稳健平滑和
短时保持提高灵敏度与稳定性，但不会把硬件不存在的信息包装成精确 3D 定位；需要时
应再接视觉人脸方向消除前后歧义。

## 验证

```bash
source .venv/bin/activate
ruff check .
ruff format --check .
pytest
reachy-mini-app-assistant check .
python scripts/probe_realtime.py --seconds 5
```

上机验收重点：

1. 连续对话时没有旧输入突发补发，首段下行音频立即播放；
2. 播放期间持续说话约 100 ms 后，扬声器立即清空，旧 response 不再复活；
3. 只播放机器人语音时不会触发插话；
4. 动作循环维持 50 Hz，网络抖动和 JPEG 编码不造成运动卡顿；
5. 左右说话位置改变时头部方向稳定跟随，静音后平滑回中。

协议依据：

- [MiniCPM-o Realtime API overview](https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/)
- [MiniCPM-o Video full-duplex](https://minicpmo45.modelbest.cn/docs/en/realtime-api/video/)
- [MiniCPM-o Realtime examples](https://minicpmo45.modelbest.cn/docs/zh/realtime-api/examples/)
- [Reachy Mini Python SDK](https://huggingface.co/docs/reachy_mini/SDK/python-sdk)
- [Reachy Mini sound DoA example](https://github.com/pollen-robotics/reachy_mini/blob/950c29eacfedd439595f7b62e9ae60f27c9096d4/docs/source/examples/sound_doa.md)
- [Reachy Mini contributor instructions](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md)
