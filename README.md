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
- 相机持续保留最新 640 px JPEG、最多 1 fps；默认每 2 秒随音频发送一帧，降低视觉
  prefill；网络拥塞时只保留最新输入，不补发旧音频。

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

其余可调项见 [.env.example](.env.example)。协议采样率、1 秒上行音频单元、640 px
视觉宽度、50 Hz 动作和 300 秒会话上限是刻意固定的。MiniCPM-o 4.5 的模型配置与
官方浏览器链路都采用 1 秒音频时基；YRobot 在单元完成后立即发送，不再叠加第二个
1 秒发送定时器。短分块只适合作为 probe A/B 项，不能默认假定它会降低延迟。

## 运行结构

```text
XVF3800 mic ─┬─ 20 ms VAD + echo guard ── local barge-in ── clear_player()
             └─ exact 1 s F32LE units ─── MiniCPM-o Realtime
camera ───── latest JPEG, default send @ 0.5 fps ─┘

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

## 延迟日志与调优

默认值优先压低本地排队：播放预卷为 `0 ms`、播放器最多 `2` 个 buffer、插话检测
约 `80 ms`。日志会分别给出音频单元完成到发送、服务端 prefill / generate / wall、
首段音频和播放器清空耗时。`session.created` 之前的等待是建连时间，不是每轮对话延迟；若
`server wall` 持续超过 1 秒，服务端会跟不上实时输入。已有 `.env` 会覆盖这些新
默认值，请同步 [.env.example](.env.example) 中对应的低延迟配置。

生产默认保持 1 秒音频单元。视觉帧在协议中是可选的，可将
`YROBOT_VISION_SEND_INTERVAL_SECONDS` 从默认 `2` 调大到 `3..10`，减少视觉
prefill；相机仍以 1 fps 更新最新画面，只有上行发送降频。调到 `1` 即每个音频单元
都带一帧。若听到首音频偶发断续，可小幅增加 `YROBOT_PLAYBACK_PREROLL_MS`，代价是
最多增加相应的首包播放等待和打断尾音。

## DoA 的物理边界

Wireless 麦克风阵列只返回水平角，`0=左、π/2=正前/正后、π=右`。线阵本身无法仅靠
DoA 区分正前和正后，也无法估计俯仰角。YRobot 会用本地近端语音门控、稳健平滑和
短时保持提高灵敏度与稳定性，但不会把硬件不存在的信息包装成精确 3D 定位；需要时
应再接视觉人脸方向消除前后歧义。

DoA 不经过 MiniCPM WebSocket。YRobot 按 Reachy 官方示例轮询当前 daemon 的
`/api/state/doa`，由 daemon 统一执行 XVF3800 USB 控制读取。LOCAL media 仍会按
SDK 1.9 的实现创建未使用的 `AudioDoA` wrapper，但应用进程不再调用
`media.get_DoA()` 发起第二路控制读取。连接失败（含 connect timeout）、500、`null`
或坏响应会指数退避，方向 tracker 在短暂故障期间继续按原有 hold / decay 平滑释放。
轮询采用自适应频率：空闲约 `2 Hz`，检测到近端人声后升到
`YROBOT_DOA_HZ`（默认 `10 Hz`）；机器人播放且没有近端人声时不读取。健康链路下，
近端 VAD 触发后最多一个活动周期内恢复。

daemon 的 DoA handler 会同步执行 USB control transfer；客户端取消 HTTP 请求不能取消
已经进入 daemon 的底层 USB 读取。因此单次 read timeout，或连续三次超过 `30 ms` 的
慢读取，会为本次 YRobot 进程熔断 DoA，避免重复请求拖住同一 daemon 的动作控制事件
循环。修复设备或重启 daemon 后再重启 YRobot 即可恢复。

若 daemon 持续返回错误，检查是否还有其他进程直接读取麦克风控制端点，并确认
XVF3800 音频固件至少为 `2.1.0`（建议使用当前 Reachy 发行版配套固件）。不要在
YRobot 内 USB reset 或重建整个媒体栈，这会同时打断录音和播放。

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
2. 播放期间持续说话约 80 ms 后，扬声器立即清空，旧 response 不再复活；
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
