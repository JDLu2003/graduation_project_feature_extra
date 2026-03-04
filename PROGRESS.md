# 进度说明

## 2026-3-04

- [x] 初始化 git仓库（初始化 git 仓库，补充 git 规则要求，补充忽略的文件）
- [x] 重构 config.yaml 结构，实现提取器配置的解耦，并更新 src/config.py 以匹配新结构。为 config.yaml 添加详细注释。
# 进度说明

## 2026-3-04

- [x] 初始化 git仓库（初始化 git 仓库，补充 git 规则要求，补充忽略的文件）
- [x] 重构 config.yaml 结构，实现提取器配置的解耦，并更新 src/config.py 以匹配新结构。为 config.yaml 添加详细注释。
- [x] 实现 dev.txt 解析器（UtteranceRecord / PersonEntry），其验证通过内部断言完成。
- [ ] 实现上下文索引构建器，用于非说话人视频查找
- [ ] 实现视频帧采样器，支持 uniform/middle/first 策略
- [ ] 定义抽象 FeatureExtractor 基类 (OCP 接口)
- [ ] 实现 512→1024 线性投影模块
- [ ] 实现 CLIPVisualExtractor，包含帧聚合和 L2-norm
- [ ] 实现端到端 pipeline 编排，包括保存/验证循环
- [ ] 添加 main.py 入口点，支持 argparse 配置注入
- [ ] 实现上下文索引构建器，用于非说话人视频查找
- [ ] 实现视频帧采样器，支持 uniform/middle/first 策略
- [ ] 定义抽象 FeatureExtractor 基类 (OCP 接口)
- [ ] 实现 512→1024 线性投影模块
- [ ] 实现 CLIPVisualExtractor，包含帧聚合和 L2-norm
- [ ] 实现端到端 pipeline 编排，包括保存/验证循环
- [ ] 添加 main.py 入口点，支持 argparse 配置注入
