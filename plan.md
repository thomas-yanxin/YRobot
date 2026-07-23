# YRobot MiniCPM-o 4.5 实时交互修复计划

状态：代码修复与自动化验证已完成；等待 Reachy Mini Wireless 真机体验复测。

## 已确认环境

- 目标硬件：Reachy Mini Wireless（CM4）。
- MiniCPM-o Realtime Gateway：
  `wss://10.0.16.184:8006/v1/realtime?mode=video`。
- 应用类型：需要本机媒体与确定性动作控制循环的 Python 应用。
- Reachy Mini SDK：1.9.0。
- 服务端采用 OpenBMB/MiniCPM-o-Demo Realtime API。

## 用户目标

1. 降低用户说完到首个可闻回复的端到端延迟。
2. 插话后立即物理静音，任何旧回答音频都不能恢复；机器人自身播放不能误触发插话。
3. 动作控制保持固定频率、单一 owner、平滑且拟人。
4. 修复音视频输入、会话状态与回复 turn 错配导致的答非所问。
5. 在协议的 300 秒 video session 上限下自动、干净地重建会话并持续服务。

## 技术方案

- 严格保持 URL mode、视频上传行为、session 生命周期三者一致。
- 使用有界、带 epoch 的输入/播放队列；session 或插话切换时原子失效旧数据。
- 音频采集、摄像头、DoA、播放、WebSocket 和动作控制各自隔离，慢设备调用不阻塞音频热路径。
- 用短启动缓冲和绝对播放时钟降低首响并避免音频 underrun。
- 使用 AEC 后 VAD，加播放参考相关性防护；插话立即清空应用队列和 GStreamer player。
- 显式处理 `session.closed`、WebSocket close、close acknowledgement、超时与带抖动重连。
- 动作由唯一 50 Hz 控制器合成，DoA 独立采样，所有轴限速/限加速度。
- 补齐协议、并发、音频 epoch、回声保护、重连和动作周期回归测试。

## 阻塞问题

无。用户已明确硬件、Gateway、目标模式和需要直接修复当前应用。

## 验收

- 自动化：81 个 pytest、ruff、compileall、依赖检查与无硬件集成测试通过。
- Gateway：真实 WSS 探针已验证 500 ms / 8000-sample 音频单元、queue/init、
  `force_listen`、listen metrics 和干净关闭；当前 session init 约 6.3 秒，
  input-to-listen 约 130 ms。
- 真机：插话静音、旧音频不恢复、自声不误触发、首响延迟、动作连续性及至少两次
  video session rollover 需要在 Wireless 实机上复测。
