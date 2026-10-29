# AL-Duet · 状态机驱动的多 AI 同场社交行为平台

这是一个**状态机驱动**的项目：每个 AI 角色的大脑是一套 70 状态社交 FSM +
情绪 FSM（数据驱动，`engine/social/data/states.yaml` + `rules.yaml` 随仓库
分发）。感知只负责喂事实，**决策全部由状态机做出**——单人环境就进探索模式
（航点巡游 + 防撞），见到人就切入社交（接近 → 问候 → 对话 → 告别；个人
空间被侵入时后撤，恢复后再互动）——然后 ARDY 冻结扩散模型把每一个决策
执行成连续步态。角色看起来"有社交行为"，是因为状态机在驱动，不是动作库。

N 个角色在同一三维街景里各自运行这条闭环，支持两种推理结构：

* **非实时（推理后播放）**：整个 episode 离线生成写入 `web/data/episode.json`，
  前端按 20 fps 纯播放，渲染永不卡顿；
* **实时并行**：N 个角色共享一份 ARDY 权重，每个仿真窗所有角色在**一次
  扩散 pass** 里同时生成（`engine/ardy.window_batch`），帧即时推送到后端
  直播缓冲，页面随产随播。

本目录与 `ARDY-V/`（视觉驱动的第一人称导航）同一 ARDY 运动生成底座，
驱动方式不同、相互独立、并行封装。

## 管线

```
感知（几何真值 | 页面截图+YOLOv8m）
        │ 事实：可见/距离/朝向/话轮/事件
        ▼
社交 FSM（70 状态，稀疏决策：dyad 1.2 s / solo 2.0 s，事件即触发）
        ＋ 情绪 FSM（评价 → 心境 → 调制阈值与距离偏好）
        │ 行为计划：approach / greet / converse / retreat / explore
        ▼
ARDY Core8（冻结，steps=10，N 角色一次扩散 pass）
        │ 姿态（模型只出姿态——无 root/heading/hand 约束、无稳定器）
        ▼
外部根控制器：世界 root 逐帧跟随原始 chunk 位移（按净行进方向对齐朝向，
零滑步）；步态打断蠕变接近（落点精度 ~0.15 m）；站立漂移用模型自产
stand 快照重置 history + 0.4 s 关节交叉淡化
        ▼
VRM 重定向 ──► 实时直播缓冲（播放头锁帧）或 episode 落盘
```

状态机与控制器的完整调优经验（过冲/滑步/极限环的病理与修复）写在
`engine/actors.py`、`engine/explore.py` 的注释里。

## 仓库结构

```
AL-Duet/
├── engine/            贯通脚本（智能全在这里）
│   ├── run.py         python -m engine.run — 视觉→状态机→ARDY→世界→推送/落盘
│   ├── ardy.py        ARDY Core8 封装：共享模型、批式并行窗口、stand 重置
│   ├── actors.py      Agent：世界变换（窗内插值转向）+ 根跟随控制器
│   ├── explore.py     非社交探索运行器（waypoint + 防撞 + 避让）
│   ├── vision.py      感知回路：geometric（世界真值）| render（截图+YOLOv8m）
│   └── social/        70 状态社交 FSM + 情绪 FSM + 数据（states/rules yaml）
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

## 状态机行为要点（实测校准）

* 接近：走 2 窗 + 立 1 窗的步态打断蠕变，停距 1.0 m，问候精准落在
  1.0–1.1 m 社交距离，不触碰 0.6 m 个人空间安全线；
* 问候 → 对话：`greet_initiate/greet_ack` 话轮由 floor 策略保证同一时刻
  只有一人发声；对话可持续数分钟稳定轮转（speak/listen/backchannel）；
* 个人空间：`g_space_violation`（d<0.6 m，硬优先级）触发后撤——撤退豁免
  步态打断并连续行走，真正把距离拉开，恢复后可再次互动；
* 模式切换：6 s 无人可见且无进行中的话轮 → 回探索模式；见到人立即切入
  社交。探索运行器负责边界反弹、道具避让与对他人身体的软斥力。

## 已知限制

* 感知默认 `geometric`（世界真值距离/朝向），`render` 才是像素级视觉；
  episode meta 里诚实标注所用模式。
* 避障是"边界矩形 + 道具圆列表"（`engine/run.py: WORLD_BOUNDS/PROPS`），
  不是网格级碰撞；换场景需要调整。
* 无物理本体（MuJoCo 物理在另一条内部管线里）；N>2 时 FSM 观察最近的
  邻居；语音是文本台词（HUD 展示），无 TTS。
* 演示 episode 由内部策略生成，仅供快速查看渲染效果。

## 许可

代码 MIT（见仓库根 LICENSE）。`assets/avatar.vrm` 是第三方演示角色模型
（Mafuyu v0.91 · 有坂みと），**其许可不属 MIT**、由原权利人保留，仅作
演示随仓库分发；请替换为自己的 VRM 用于任何正式用途。樱花街景模块来自
MIT 的 sakura-crossing 项目（credit 见 `assets/scenes/*/scene.json`）。
