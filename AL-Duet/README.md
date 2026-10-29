# AL-Duet · 多 AI 同场社交行为平台

把冻结的运动扩散模型（ARDY Core8）接到一套**社交/情绪状态机**上：N 个 AI
角色在同一三维街景里各自运行"感知 → 社交决策 → ARDY 连续步态生成 → 外部
根控制器 → 渲染"的闭环。角色**自己判断**环境里有没有别人——单人时进
探索模式（航点巡游 + 防撞边界），见到人自动切入社交状态机（接近 → 问候 →
对话 → 告别 / 个人空间被侵入时后撤），全程动作都是模型逐窗生成的，没有
手写动画。

支持两种推理结构：

* **非实时（推理后播放）**：整个 episode 离线生成写入 `web/data/episode.json`，
  前端按 20 fps 纯播放，渲染永不卡顿；
* **实时并行**：N 个角色共享一份 ARDY 权重，每个仿真窗所有角色在**一次
  扩散 pass** 里同时生成，帧即时推送到后端直播缓冲，页面随产随播。

本目录与 `ARDY-V/`（第一人称视觉导航）同一运动生成底座，独立封装、互不依赖。

## 仓库结构

```
AL-Duet/
├── engine/            贯通脚本（智能全在这里）
│   ├── run.py         python -m engine.run — 视觉→状态机→ARDY→世界→推送/落盘
│   ├── ardy.py        ARDY Core8 封装：共享模型、批式并行窗口、stand 重置
│   ├── actors.py      Agent：世界变换（逐帧插值转向）+ 根跟随控制器
│   ├── explore.py     非社交探索运行器（waypoint + 防撞 + 避让）
│   ├── vision.py      感知回路：geometric（世界真值）| render（截图+YOLOv8m）
│   └── social/        70 状态社交 FSM + 情绪 FSM（states.yaml/rules.yaml）
├── backend/           FastAPI：静态服务 + N-agent 直播帧缓冲 + HUD 推送
├── web/               three.js 播放器：场景 + N 个 VRM + 播放/实时双模式
│   ├── vendor/        three.js / three-vrm（MIT）
│   ├── scenes/        樱花街景模块（MIT）
│   └── data/          一段预生成演示 episode
├── assets/            演示 VRM + 场景清单
└── requirements.txt
```

## 运行

```bash
cd AL-Duet
# 外部依赖：ARDY 运行时（见下）——先设环境变量
set ALDUET_AL0A=D:\path\to\AL-0A

python -m backend --port 8840            # 平台服务

# 非实时：推理后播放（默认 geometric 感知，无浏览器）
python -m engine.run --mode offline --actors 2 --windows 80
# 浏览器打开 http://127.0.0.1:8840/  → 播放 episode

# 实时：随产随播（--realtime 按 0.4 s 仿真节拍 pacing；--windows 0 无限跑）
python -m engine.run --mode live --realtime --windows 0 --actors 2 --join-after 20
# 浏览器打开 http://127.0.0.1:8840/?live=1
#   角色先独自探索，第二个角色 20 s 后加入，双方见到即切入社交模式

# 更多的 agent（一次扩散 pass 批式生成）
python -m engine.run --mode offline --actors 3 --windows 120

# 实时 + 真实视觉闭环（页面第一人称捕获 + YOLOv8m 检测 + 深度射线测距）
python -m engine.run --mode live --vision render --windows 300
```

仓库自带一段预生成双人 episode（`web/data/episode.json`），装好依赖后
`python -m backend --port 8840` 打开首页即可离线播放，不需要 ARDY 运行时。

## 外部依赖（不在本仓库内分发）

* **ARDY 模型实现 + Core8 checkpoint**：NVIDIA 的 ARDY 发布版
  （`ardy` 包，checkpoint `ARDY-Core-RP-20FPS-Horizon8`）。sha256 已核对
  与官方一致。注意 Core8 训练时 `num_base_steps: 10`，采样只支持 2–10 步。
* **`ALDUET_AL0A`**：指向携带 `app/ardy_runtime.py`（ArdyGenerator 运行时
  封装）与 `runs/v7_clip/text_emb`（LLM2Vec 4096 维文本嵌入）的检出。
* Python 依赖见 `requirements.txt`；`--vision render` 另需 ultralytics 与
  playwright。

## 行为与实现要点

* 动作 = 审计后的干净路径：ARDY 只出姿态（无 root/heading/hand 约束、无
  姿态稳定器——那些是早期畸变来源），世界 root **逐帧跟随原始 chunk 自身
  位移**（按净行进方向对齐朝向），结构上不可能滑步/倒滑；
* 站立自回归漂移用模型自产的 stand 快照重置 history，并做 0.4 s 关节
  交叉淡化（不弹跳）；转向在窗内 8 帧线性插值（不瞬转）；
* 接近用"走 2 窗 + 立 1 窗"的步态打断蠕变，落点精度 ~0.15 m，问候精准
  落在 1.0–1.1 m 社交距离，不触碰 0.6 m 个人空间安全线；
* 直播播放头锁帧时间戳（落后缓冲 0.2 s、1.5x 渐进追赶），轮询带重入锁，
  不因网络抖动瞬移。

## 已知限制

* 感知默认 `geometric`（世界真值距离/朝向），`render` 才是像素级视觉；
  episode meta 里诚实标注所用模式。
* 避障是"边界矩形 + 道具圆列表"（`engine/run.py: WORLD_BOUNDS/PROPS`），
  不是网格级碰撞；换场景需要调整。
* 无物理本体（MuJoCo 物理在另一个内部管线里）；N>2 时 FSM 观察最近的
  邻居；语音是文本台词（HUD 展示），无 TTS。
* 演示 episode 由内部策略生成，仅供快速查看渲染效果。

## 许可

代码 MIT（见仓库根 LICENSE）。`assets/avatar.vrm` 是第三方演示角色模型
（Mafuyu v0.91 · 有坂みと），**其许可不属 MIT**、由原权利人保留，仅作
演示随仓库分发；请替换为自己的 VRM 用于任何正式用途。场景模块来自
MIT 的 sakura-crossing 项目（credit 见 `assets/scenes/*/scene.json`）。
