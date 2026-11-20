# 第三方来源与许可

以下条目明确根软件许可的例外，不替代上游完整条款或全部分发义务。

- 两条路线 web/vendor/ 下的 three.js、three-vrm 等第三方库保留上游许可；原文位于 [ARDY-V three](ARDY-V/web/vendor/three/LICENSE)、[ARDY-V three-vrm](ARDY-V/web/vendor/three-vrm/LICENSE)、[AL-Duet three](AL-Duet/web/vendor/three/LICENSE)、[AL-Duet three-vrm](AL-Duet/web/vendor/three-vrm/LICENSE)。cannon-es 等其他外部库也按其实际来源版本的许可使用。
- sakura-crossing 场景：保留 [assets 原许可](AL-Duet/assets/scenes/sakura_crossing/LICENSE)、[web 原许可](AL-Duet/web/scenes/sakura_crossing/LICENSE) 及相应 scene.json 中的 credit。
- ARDY 外部生成器、DINOv2、VRM 角色、训练数据及其他模型和素材，各自保留代码、权重和资源条款；不由根 LICENSE 统一授权。角色模型仍须由使用者合法自备。
- 历史主项目 MIT 授权保持有效，见 [原文](licenses/legacy/MIT.txt) 和 [版本边界](LICENSING.md)。

## YOLO 权重与集成：独立授权边界

ARDY-V/checkpoints/yolo26n-depth.pt 内嵌声明为 AGPL-3.0 (https://ultralytics.com/license)，
序列化模型引用 ultralytics.nn.tasks.DepthModel 等 Ultralytics 模块；不能将其统称为可由本项目独立授权的读出头。
静态核查文件的 SHA-256 为 befed1b8561d8b2eaa66274b070cdfc9c44853bda6d6d62a04759f9383af74e9。
此结论仅针对该文件，不推定所有 checkpoint 使用相同许可。

相关集成位于 ARDY-V/ardy_v/yolo_depth.py、ARDY-V/tools/train_yolo_depth.py 和 AL-Duet/engine/vision.py。
这些既有文件与权重不会因本次声明丧失原有权利；本项目的非商业许可或商业协议不覆盖 Ultralytics 的独立权利。
遵循 AGPL 的分发、修改和组合使用须履行适用义务，不能以本项目非商业条款取代或限制其授予的权利。
计划商业交付时，应取得覆盖实际用途与分发范围的上游授权，或排除并替换相关实现和权重、验证构建产物。
本说明不是已完成 AGPL 合规或已持有上游商业授权的声明。
