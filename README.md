# FlowState-Embodied

研究仓库：把冻结的运动扩散模型（ARDY Core8）接到第一人称视觉导航上——
相机画面经冻结 DINOv2 编码出公制深度与碰撞掩码，在线累积占据地图，A* 规划
连续路线，再由 ARDY 作为根边界条件逐窗生成步态，驱动 three.js 里的 VRM 角色。
只给起点和终点，角色自己判断碰撞、规划路线、发现环境变化后重规划。

本仓库开源这条路线（视觉 → 占据 → 规划 → 生成 → 重定向）的实现，
以及可复跑的**部分测试路线**（`ARDY-V/routes/`）。生成器本体属于外部研究
代码，不在本仓库内分发。

## 仓库结构

```
FlowState-Embodied/
├── README.md          本文件（仓库说明）
├── LICENSE            MIT
├── .gitignore
└── ARDY-V/            项目主体，独立目录封装
    ├── ardy_v/        服务、运行时、导航、推理、训练
    ├── web/           three.js + three-vrm + cannon-es 前端
    ├── routes/        测试路线 JSON
    ├── tools/         路线运行器、数据生成、训练、QC 探针
    ├── checkpoints/   训练好的读出头
    ├── assets/        演示 VRM
    ├── docs/          演示截图
    └── requirements.txt
```

下文所有命令默认先 `cd ARDY-V`。

```
第一人称 RGB ──► DINOv2 ViT-B/14（冻结）+ 深度头 + 碰撞头（本仓库训练）
                     │
                     ▼
            在线占据地图（世界系累积，灰度可视化）
                     │  A* + Chaikin 平滑 ──► 连续路线
                     ▼
            RouteTracker ──► 每窗 8×3 机体坐标系根节点路径
                     │        （航向整形：≤0.35 rad/窗 闭合，
                     │         窗内 ±0.7 rad 裁剪，单调前进）
                     ▼
            ARDY Core8（冻结，6 步 DDIM，generate_path，stride=4 关键帧）
                     │
                     ▼
            VRM 重定向（three-vrm）+ cannon-es 物理（箱子可拖拽）
```

## 这条路线为什么这样设计

- **根节点边界而不是逐关节控制**：ARDY 的原生接口是根轨迹/关键帧/末端等式约束。
  每 8 帧给模型一条机体坐标系的根路径（前向弧长、横向偏移、绝对航向偏移），
  身体动作由模型自己生成，不存在"根被后处理拉走、脚底打滑"的问题。
- **关键帧 stride=4**：把根/航向/高度约束到窗口的每一帧上会过度约束，
  上身会在约 12 个窗口内逐渐塌陷（头掉到骨盆以下）。稀疏关键帧（stride=4）
  保持同样的路径跟踪误差（0.032 m），姿态不塌。这是仓库里最重要的一条经验。
- **指令整形**：重规划风暴曾每窗下发 ±0.7 rad 的交替航向跳变和横向漂移，
  离线重放这些指令即可复现弯腰。现在追踪器按 ≤0.35 rad/窗闭合航向差，
  窗内逐帧航向偏移裁到 ±0.7 rad（生成器训练分布内），位置从整形后的航向
  重新积分，保证位置/航向约束自洽。
- **在线占据而非一次性建图**：掩码按相机位姿投影到世界系、带 hit/miss 更新；
  箱子被拖拽或路线被挡时，地图失效触发重规划（见 `routes/blocked_midway.json`）。

## 测试路线（routes/）

路线是纯 JSON：起点固定、目标、箱子摆位、超时、验收条件。`tools/run_route.py`
用与用户完全相同的页面 API（setTarget / moveBox / reset）跑真实闭环（需要
服务在跑 + GPU），结果写到 `runs/routes/<name>.json` 并截图。

```bash
python tools/run_route.py routes/direct.json          # 直达（中央箱需绕行）
python tools/run_route.py routes/blocked_midway.json  # 中途放箱 → 必须重规划
python tools/run_route.py routes/slalom.json          # 三箱交错绕行
```

| 路线 | 内容 | 实测（2026-10-29，本机） |
|---|---|---|
| `direct` | (-5,-5) → (5,5)，中央固定箱 | **通过** 26.2 s · 7 次重规划 · minSdf 1.78（同路线首跑曾因地图误报碰撞失败，见局限） |
| `blocked_midway` | 行走中在前方 2.5 m 视野内放箱 | **通过** 放箱后 0.6 s 检出失效 · 30.2 s 到达 · 9 次重规划 · minSdf 0.98 |
| `slalom` | 三箱交错在对角线上 | **通过** 20.1 s · 7 次重规划 · minSdf 2.10 |

`minSdf` 是执行根轨迹到障碍的最小有符号距离（负值为侵入）；
`collision` 是物理碰撞，如实报告、不隐藏。

## 训练了什么 / 冻结了什么

- **冻结**：ARDY Core8（`ARDY-Core-RP-20FPS-Horizon8`）、DINOv2 ViT-B/14 reg4。
- **本仓库训练的读出头**（checkpoints 内）：绝对深度（`depth.pt`）、
  碰撞掩码（`boundary.pt`）、在线导航融合头（`nav.pt`，留出集 absrel 0.171 /
  掩码 IoU 0.866）、YOLO 特征深度（`yolo26n-depth.pt`，备选）。
- 没有 ARDY LoRA / 精调；动作质量全部来自冻结模型本身。

## 运行 demo

```bash
pip install -r requirements.txt
python -m uvicorn ardy_v.service:app --host 127.0.0.1 --port 8830
# 打开 http://127.0.0.1:8830/
```

- 点地面设目标；角色规划、行走、到点站立。
- 箱子有刚体物理、可用鼠标拖拽；导航器把它融进占据地图，路线被挡或出现
  更优路线时重规划。
- HUD：第一人称画面、预测深度（灰度）、在线占据地图、逐窗导航状态。

## 外部依赖（不在本仓库）

| 依赖 | 来源 | 环境变量 |
|---|---|---|
| ARDY Core8 生成器（`run_r07_ardy_precision` / `run_r10_ardy_inverse`） | 作者 AL 研究树——需自备同接口的动作生成栈 | `FLOWSTATE_AL0A`、`FLOWSTATE_CIP_RUNS` |
| VRM 角色 | 仓库内附 `assets/avatar.vrm` 仅作演示；也可换任意 VRM 0/1 人形 | `FLOWSTATE_VRM` |
| DINOv2 ViT-B/14 reg4 权重 | 首次运行由 torch.hub 缓存自动下载 | — |
| YOLO 权重（可选，仅 `tools/train_yolo_depth.py`） | 自行下载（AGPL 许可） | `FLOWSTATE_YOLO_WEIGHTS` |

## 目录结构（ARDY-V/ 内）

```
ardy_v/       服务（FastAPI）、ARDY 运行时、路线追踪、占据地图、
              深度/导航/YOLO 推理、数据与训练 IO
web/          three.js + three-vrm + cannon-es 前端（vendor 内置）
routes/       测试路线 JSON（本仓库开源的测试基准）
tools/        run_route.py 路线运行器；数据生成、训练、QC 探针
checkpoints/  训练好的读出头（depth / boundary / nav / yolo-depth）
assets/       演示用 VRM 角色
docs/         演示截图
```

## 已知局限

- 导航器的占据地图在个别接近角下会误报障碍（深度头的眼高域偏移），
  偶发触发多余重规划；地图误修复原后重规划风暴显著减少，但**间歇性失败
  仍然存在**（`direct` 路线首跑即因误报碰撞失败、复跑通过）。修复方向是
  用约 1.3 m 眼高的第一人称数据重训深度/掩码头。
- 本仓库仅为"视觉 + 规划 + 重定向"半边；没有外部 ARDY 栈无法生成动作。
- Core27 无手指链，操作类任务不在范围内。

## 素材与许可

代码以 [MIT](LICENSE) 发布。`web/vendor` 内第三方库（three.js、three-vrm、
cannon-es）保留各自许可证。`assets/avatar.vrm` 为第三方角色模型，
**仅用于本仓库的演示与评测**，不随 MIT 许可授权，二次分发/商用请自行确认权利。
`checkpoints/*.pt` 为本仓库训练的读出头，按原样提供。
