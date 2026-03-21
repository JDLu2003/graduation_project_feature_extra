# 进度说明

## 2026-3-04

- [x] 初始化 git仓库（初始化 git 仓库，补充 git 规则要求，补充忽略的文件）
- [x] 重构 config.yaml 结构，实现提取器配置的解耦，并更新 src/config.py 以匹配新结构。为 config.yaml 添加详细注释。
- [x] 实现 dev.txt 解析器（UtteranceRecord / PersonEntry），其验证通过内部断言完成。
- [x] 创建 .gitignore 文件。
- [x] 创建 src/context_index.py 文件骨架，定义 build_context_index 函数原型。
- [x] 创建 src/video_utils.py 文件骨架，定义 sample_frames 函数原型。
- [x] 创建 src/extractors/base.py 文件骨架，定义 FeatureExtractor 抽象基类。
- [x] 创建 src/extractors/projection.py 文件骨架，定义 LinearProjection PyTorch nn.Module 类骨架。
- [x] 创建 src/extractors/clip_extractor.py 文件骨架，定义 CLIPVisualExtractor 类骨架。
- [x] 整合 main.py，在 run_pipeline 中调用所有模块的接口原型，包括输出模块的接口。
- [x] 实现上下文索引构建器，用于非说话人视频查找 (已验证)
- [ ] 实现视频帧采样器，支持 uniform/middle/first 策略 (具体实现)
- [x] 实现 512→1024 线性投影模块 (已整合到 CLIPVisualExtractor)
- [x] 实现 CLIPVisualExtractor，包含帧聚合和 L2-norm (已验证)


## 2026-03-04 - 添加日志打印和文件结构重构

- **任务描述**: 在 pipeline 和关键模块中添加打印信息，提升可观测性。同时，进行了文件结构重构。
- **完成内容**:
    - 在 `src/extractors/visual_clip/clip_encoder.py` 中添加了 CLIP 模型加载和空帧警告日志。
    - 在 `src/extractors/visual_clip/strategy.py` 中添加了非说话人上下文视频查找和视频文件缺失警告日志。
    - 在 `src/pipeline.py` 中添加了处理每个 utterance 的日志。
    - 在 `src/saver.py` 中取消了跳过现有文件时的日志注释。
    - 对项目文件结构进行了重大调整，包括添加新的模块文件，并删除旧的模块文件。
- **Git 提交**: `feat: 添加日志打印，提升 pipeline 可观测性` (`6bebf23`)
- **验证**: 运行 `conda run -n security python main.py --config config.yaml` 命令，确认所有预期日志输出均已正确显示。
- **后续计划**: 继续根据 CLAUDE.md 中的指导原则进行下一步的特征提取实现。

## 2026-03-21

- [x] 新增 `scripts/stat_role_frequencies.py`，复用 `src/parser.parse_dev_txt` 统计角色在对话中的出现频次。
- [x] 输出角色的四项统计：对话中出现次数、语句中出现次数、作为说话人的次数、作为非说话人的次数。
- [x] 增加可选 `CSV` / `JSON` 导出，并支持按语句次数、对话次数、说话次数、听众次数或名称排序。
- [x] 运行脚本完成验证，确认可直接读取现有 `config.yaml` 和 `dev.txt`。
- [x] 新增 `scripts/extract_faces_with_ffmpeg_mtcnn.py`，完成 `ffmpeg` 抽帧 -> `MTCNN` 画框 -> 视频重建的独立流程。
- [x] 脚本输出统一落在 `logs/<video_stem>_mtcnn_<timestamp>/` 下，包含原始帧、框图、重建视频和检测清单。
- [x] 完成脚本语法与 `--help` 验证；当前执行环境未安装 `facenet_pytorch`，脚本已改为延迟导入并给出明确报错。
- [x] 按要求补齐音频保留逻辑，并将人脸置信度阈值默认设为 `0.95`，可通过参数覆盖。
- [x] 在主仓库新增 `Makefile`，统一 `role-stats / face-box-video / talknet` 相关命令入口。
- [x] 调整 `src/extractors/base.py` 与 `src/extractors/face_scene_fr/strategy.py` 的类型注解与文档说明。
- [ ] 整理 `talknet_asd` 与 `audio_opensmile` 的子模块迁移配置（目标：在新机器可直接 `git submodule update --init --recursive`）。
