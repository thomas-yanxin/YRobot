# YRobot 重构计划：不可变 MiniCPM-o 服务端

状态：客户端实现与自动验证完成；剩余工作只有 Reachy Mini Wireless 实机验收。

## 1. 硬约束

- 应用运行在 Reachy Mini Wireless CM4。
- 唯一模型入口是
  `wss://10.0.16.184:8006/v1/realtime?mode=video`。
- Gateway、Python worker 和 comni C++ backend 均不可修改、重启配置或部署补丁。
- video session 固定最多 300 秒；单 worker 不能并行预热下一会话。
- 实测 `session.init → session.created` 为 9.2–15.8 秒。这是服务端固定等待，客户端不把它
  计入可优化延迟，也不承诺消除。
- 当前 comni 只能安全使用空 `system_prompt`；YRobot 始终沿用服务端内置 persona。
- 公开协议输入固定为约一秒、16 kHz mono float32；输出为 24 kHz mono float32。

## 2. 客户端目标

只优化 Reachy 侧可控链路：

| 路径 | 目标 |
|---|---:|
| audio delta → 第一帧 speaker push | p95 ≤ 120 ms |
| VAD onset → 本地物理静音 | p95 ≤ 100 ms |
| VAD onset → force-listen control unit 出队 | p95 ≤ 100 ms |
| 播放帧 cadence | 20 ms，不累积 push 调用耗时 |
| 动作控制 | 单 owner，50 Hz |
| 视频采集对音频阻塞 | 0 |

服务端首响、300 秒会话上限和重新初始化窗口仅记录，不伪装为客户端改善。

## 3. 架构

```text
Reachy mic / XVF AEC
  -> 20 ms VAD + residual echo guard
  -> bounded one-second uplink
  -> MiniCPM-o Realtime Gateway

latest-only JPEG --------------------^

24 kHz model audio
  -> streaming 24k→16k resampler
  -> epoch playback queue
  -> absolute 20 ms sample clock
  -> Reachy local GStreamer player

DoA @ 10 Hz -> latest target -> single motion owner @ 50 Hz
```

代码只保留五个运行边界：

- `config.py`：固定服务能力与少量环境配置；
- `realtime.py`：协议、关闭栅栏、重连和空闲 rollover；
- `audio.py`：采集、VAD、自声保护、打断、重采样和播放；
- `motion.py`：DoA 与唯一动作控制器；
- `main.py`：Dashboard/CLI 生命周期。

## 4. 插话

1. 对 Reachy AEC 后的 channel 0 运行 20 ms WebRTC VAD。
2. 两个连续 voiced frame 确认用户 onset。
3. 原子递增 playback epoch、丢弃旧 response，并请求
   `mini.media.audio.clear_player()`。
4. 当前已采集的近端开头补零到 16,000 samples，立即携带 `force_listen=true`。
5. 高相关的扬声器残余音频同时从本地 VAD 和模型上行中移除。
6. 收到 `kind=listen` 前拒绝所有晚到旧音频；旧尾音永不恢复。

服务端没有公开 cancel API，所以“机器人先闭嘴”由本地 player flush 保证，服务端确认可以稍后
到达。

## 5. 播放与动作

- 播放初始 lead 为 40 ms；按绝对 deadline 推送 20 ms frame。
- 应用、SDK 和设备队列全部有界；过期上行直接丢弃。
- 回答超过播放容量时保留连续句首并截断本轮，不拼接不连续句尾。
- DoA USB 读取与动作控制分线程，阻塞 DoA 不影响 50 Hz actuator cadence。
- 所有姿态经临界阻尼、速度/加速度和机械范围限制。
- 说话微动作使用 Reachy SDK playback-synchronised wobbling。
- `set_target` 失败时恢复内部轴状态，重连后不会出现追赶跳变。

## 6. 会话生命周期

- 等 `session.queue_done` 后才发送 `session.init`。
- 等 `session.created` 后才打开新的麦克风 epoch。
- 285 秒请求 rollover，只在服务端 listening、用户安静且本地播放排空时关闭。
- 294 秒为硬关闭边界，给固定 300 秒 watchdog 和 close acknowledgement 留余量。
- 关闭顺序固定为：本地 invalidate/flush → sender quiescent → `session.close` → 等
  `session.closed` → 等 Gateway 关闭传输并完成 Worker 回收。
- 若固定 Gateway 的回收窗口仍返回 `session_failed`/HTTP 403，使用带抖动的有界退避重试，
  不依赖服务端修改或重启。
- 新会话重新初始化期间保持 idle 姿态；不播放伪造填充语音，也不复用旧上下文。

## 7. 验证

- 68 项单元与本地集成测试通过。
- Ruff、format、`git diff --check`、compileall、wheel build 和 `pip check` 通过。
- Reachy app metadata、entry point 和 `ReachyMiniApp` 类加载通过。
- 线上三层 health 正常；最近一次 probe：
  - queue：0.9 ms
  - init：14987.5 ms
  - input-to-listen：72.9 ms
  - backend `wall_clock_ms`：1.159001

## 8. 实机验收

1. 机器人独说 30 次，不发生自我打断。
2. 双讲插话 30 次，旧音频在 100 ms 内停止且不恢复。
3. 近场、远场、侧向和机械运动噪声下分别校准 VAD/AEC。
4. 验证 DoA 朝向、天线、头部和 body yaw 无阶跃。
5. 连续运行 30 分钟，覆盖至少五次 300 秒 session rollover。
