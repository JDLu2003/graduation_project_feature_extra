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
- [ ] 实现 512→1024 线性投影模块 (具体实现)
- [ ] 实现 CLIPVisualExtractor，包含帧聚合和 L2-norm (具体实现)
