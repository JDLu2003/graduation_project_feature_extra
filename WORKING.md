# 代码框架构想与开发计划

## 1. 核心任务
针对验证集（及后续训练集）数据进行多模态（文本、视频、情绪标签）特征提取工作。
专注于特征提取与特征持久化保存，不涉及后续分析和建模代码。
所有策略、路径、维度等参数通过 `config.yaml` 进行灵活管理。

## 2. 环境与技术栈
- 编程语言：Python 3.x (严格使用 Type Hints)
- 核心框架：PyTorch
- 环境管理：Conda (`security` 环境)

## 3. 数据输入规范 (Input Specification)
数据集路径：`../data_set/`
核心处理路径：`../data_set/Viedeo_en_dev/`
结构示例:
```text
../data_set/
└── Viedeo_en_dev/
    ├── dev.txt                # 对话结构与情绪标记
    └── Video_en_dev/          # 视频文件根目录
        ├── C_18/              # 对话 18
        │   ├── C_18_U_1.mp4   # 语句 1
        └── C_1001/
            ├── C_1001_U_1.mp4 # C_{对话编号}_U_{句子编号}.mp4
```
`dev.txt` 解析规则：
- 对话头：`数字 数字` (对话ID 语句总数)，例如 `18 13`。
- 语句行：`句子编号 | 文本内容 | 说话人名字 | 说话人情绪 | 听众1名字 | 听众1情绪 | ...`
- 视频映射：`Video_en_dev/C_{对话编号}/C_{对话编号}_U_{句子编号}.mp4`

## 4. 数据输出规范 (Output Specification)
输出目标文件夹：`../feat_out/`
输出结构镜像输入视频的文件夹结构。
输出示例：`../feat_out/Video_en_dev/C_1001/C_1001_U_1.pt`

Tensor 特征维度与格式：
- 每一句话的每个人（说话人 + 非说话人）提取一个 1024 维特征向量。
- 保存为 PyTorch `.pt` 文件。
- Tensor 维度：`[N, 1024]` (N 为总人数)。
- Tensor 排序：第 0 维严格按照 `dev.txt` 中人物出现顺序排列，索引 0 为说话人，后为非说话人。
- 缺失画面处理策略：非说话人特征的填充策略由配置控制 (目前为 `context_video` 结合 `zero` fallback)。

## 5. 软件工程与框架设计
- **目录结构**：`src/` 下按功能模块拆分。
- **开闭原则 (OCP)**：设计抽象基类 `FeatureExtractor`，新增提取方法通过实现接口扩展。
- **防御性编程 (Asserts)**：在输入-处理-输出全流程中大量使用 `assert` 确保数据一致性。

# 代码框架构想与开发计划

## 1. 核心任务
针对验证集（及后续训练集）数据进行多模态（文本、视频、情绪标签）特征提取工作。
专注于特征提取与特征持久化保存，不涉及后续分析和建模代码。
所有策略、路径、维度等参数通过 `config.yaml` 进行灵活管理。

## 2. 环境与技术栈
- 编程语言：Python 3.x (严格使用 Type Hints)
- 核心框架：PyTorch
- 环境管理：Conda (`security` 环境)

## 3. 数据输入规范 (Input Specification)
数据集路径：`../data_set/`
核心处理路径：`../data_set/Viedeo_en_dev/`
结构示例:
```text
../data_set/
└── Viedeo_en_dev/
    ├── dev.txt                # 对话结构与情绪标记
    └── Video_en_dev/          # 视频文件根目录
        ├── C_18/              # 对话 18
        │   ├── C_18_U_1.mp4   # 语句 1
        └── C_1001/
            ├── C_1001_U_1.mp4 # C_{对话编号}_U_{句子编号}.mp4
```
`dev.txt` 解析规则：
- 对话头：`数字 数字` (对话ID 语句总数)，例如 `18 13`。
- 语句行：`句子编号 | 文本内容 | 说话人名字 | 说话人情绪 | 听众1名字 | 听众1情绪 | ...`
- 视频映射：`Video_en_dev/C_{对话编号}/C_{对话编号}_U_{句子编号}.mp4`

## 4. 数据输出规范 (Output Specification)
输出目标文件夹：`../feat_out/`
输出结构镜像输入视频的文件夹结构。
输出示例：`../feat_out/Video_en_dev/C_1001/C_1001_U_1.pt`

Tensor 特征维度与格式：
- 每一句话的每个人（说话人 + 非说话人）提取一个 1024 维特征向量。
- 保存为 PyTorch `.pt` 文件。
- Tensor 维度：`[N, 1024]` (N 为总人数)。
- Tensor 排序：第 0 维严格按照 `dev.txt` 中人物出现顺序排列，索引 0 为说话人，后为非说话人。
- 缺失画面处理策略：非说话人特征的填充策略由配置控制 (目前为 `context_video` 结合 `zero` fallback)。

## 5. 软件工程与框架设计
- **目录结构**：`src/` 下按功能模块拆分。
- **开闭原则 (OCP)**：设计抽象基类 `FeatureExtractor`，新增提取方法通过实现接口扩展。
- **防御性编程 (Asserts)**：在输入-处理-输出全流程中大量使用 `assert` 确保数据一致性。

## 6. 模块设计 (基于 `config.yaml` 和计划)

### `config.yaml` 核心字段 (已完成重构)
- `paths`: 输入输出路径。
- `extractor`: 指定当前活跃的特征提取器类型，例如 `active_type: "visual_clip"`。
- `visual_clip_config`: CLIP 视觉特征提取器的独立配置块。
  - `model_name`: CLIP 模型名称。
  - `device`: 计算设备。
  - `clip_output_dim`: CLIP 原始输出维度。
  - `target_dim`: 最终目标维度。
  - `frame_sampling`: 帧采样策略 (`strategy`, `num_frames`, `aggregation`)。
- `non_speaker`: 非说话人特征策略 (`strategy`, `fallback`)。
- `pipeline`: 管道控制 (`skip_existing`, `show_progress`)。

### `src/config.py` (已完成重构)
- 定义类型化的 `dataclass` (例如 `AppConfig`, `PathsConfig`, `VisualClipConfig`, `ExtractorConfig` 等) 匹配 `config.yaml` 结构。
- `from_yaml` 方法加载 YAML 文件并转换为 `AppConfig` 对象，根据 `extractor.active_type` 动态加载对应提取器配置，并进行路径存在性等基础断言。

### `src/parser.py` (已完成并验证)
- 负责解析 `dev.txt` 文件。
- 定义 `UtteranceRecord` 和 `PersonEntry` 等 `dataclass` 来存储解析结果。
- 包含严格的防御性断言，验证字段数量、格式、utterance 计数等。
- `if __name__ == "__main__":` 块已实现全面的自验证功能，包括视频文件路径的存在性检查。

### `src/context_index.py` (已实现并验证)
- 定义 `build_context_index` 函数原型，用于构建上下文视频索引。
- 模块自验证功能待实现。

### `src/video_utils.py` (已创建骨架并验证)
- 定义 `sample_frames` 函数原型，使用 `cv2` 实现 MP4 视频的帧采样功能。
- 支持 "uniform", "middle", "first" 等采样策略。
- 模块自验证功能已实现并验证。

### `src/extractors/base.py` (已创建骨架)
- 定义 `FeatureExtractor` 抽象基类，包含 `extract` 和 `output_dim` 等抽象方法。

### `src/extractors/projection.py` (已创建骨架)
- 定义 `LinearProjection` PyTorch `nn.Module`，用于将 CLIP 模型的原始输出维度 (`512`) 线性投影到目标维度 (`1024`)。
- 模块自验证功能待实现。

### `src/extractors/clip_extractor.py` (已创建骨架)
- 定义 `CLIPVisualExtractor` 类骨架，继承 `FeatureExtractor`。
- 包含加载 CLIP 模型、调用 `video_utils.py` 进行帧采样、对 CLIP 输出进行聚合、调用 `projection.py` 进行维度投影、应用 L2-norm 等逻辑。
- 模块自验证功能待实现。

### `main.py` (已完成全面整合)
- 程序入口点。
- 使用 `argparse` 处理命令行参数。
- 调用 `AppConfig.from_yaml` 加载配置。
- `run_pipeline` 函数已全面整合所有模块的接口原型，清晰展示从数据解析到特征保存的完整流程，并包含必要的防御性断言。

### 核心数据流 (已在 `main.py` 中原型化)
```
main.py
  ├─ load_config() → AppConfig
  ├─ CLIPVisualExtractor(cfg)
  │     ├─ clip.load("ViT-B/32")
  │     └─ LinearProjection(512→1024)
  └─ run_pipeline(cfg, extractor)
        ├─ parse_dev_txt() → list[DialogueRecord]
        ├─ build_context_index() → {(dialogue_id, name) → [video_paths]}
        └─ For each UtteranceRecord:
              ├─ speaker (index 0): feature extraction process
              ├─ listener (index 1..N-1): feature extraction process or zero fallback
              ├─ torch.cat → [N, 1024]
              ├─ assert shape[0]==N, shape[1]==1024
              ├─ torch.save(.pt)
              └─ load-back verify
```

## 7. 当前开发进度
- [x] 初始化 Git 仓库。
- [x] 重构 `config.yaml` 结构，将抽取器配置模块化，并更新 `src/config.py` 以匹配新结构。为 `config.yaml` 和 `src/config.py` 添加详细注释。
- [x] 实现 `dev.txt` 解析器（`src/parser.py`），其验证通过内部断言完成。
- [x] 创建 `.gitignore` 文件。
- [x] 创建 `src/context_index.py` 文件骨架，定义 `build_context_index` 函数原型。
- [x] 创建 `src/video_utils.py` 文件骨架，定义 `sample_frames` 函数原型。
- [x] 创建 `src/extractors/base.py` 文件骨架，定义 `FeatureExtractor` 抽象基类。
- [x] 创建 `src/extractors/projection.py` 文件骨架，定义 `LinearProjection` PyTorch `nn.Module` 类骨架。
- [x] 创建 `src/extractors/clip_extractor.py` 文件骨架，定义 `CLIPVisualExtractor` 类骨架。
- [x] 整合 `main.py`，在 `run_pipeline` 中调用所有模块的接口原型，包括输出模块的接口。

## 8. 下一步开发计划 (具体实现)
- [ ] 实现 `src/context_index.py` 中的 `build_context_index` 函数。
- [ ] 实现 `src/extractors/projection.py` 中的 `LinearProjection` 模块的 `forward` 逻辑。
- [ ] 实现 `src/extractors/clip_extractor.py` 中的 `CLIPVisualExtractor` 的 `__init__` 和 `extract` 逻辑。

### `src/parser.py` (待实现)
- 负责解析 `dev.txt` 文件。
- 定义 `UtteranceRecord` 和 `PersonEntry` 等 `dataclass` 来存储解析结果。
- 包含严格的防御性断言，验证字段数量、格式、utterance 计数等。

### `src/context_index.py` (已实现并验证)
- 构建上下文视频索引，用于快速查找某个对话中特定人物作为说话人的所有视频路径。
- 主要用于非说话人特征的 "context_video" 策略。

### `src/video_utils.py` (待实现)
- 使用 `cv2` 实现 MP4 视频的帧采样功能。
- 支持 "uniform", "middle", "first" 等采样策略。
- 负责视频文件的打开、帧读取、图像预处理等。

### `src/extractors/base.py` (待实现)
- 定义 `FeatureExtractor` 抽象基类，包含 `extract` (接受视频路径和人物信息，返回 `[1, 1024]` 特征) 和 `output_dim` 等抽象方法。

### `src/extractors/projection.py` (待实现)
- 实现一个 PyTorch `nn.Module`，用于将 CLIP 模型的原始输出维度 (`512`) 线性投影到目标维度 (`1024`)。

### `src/extractors/clip_extractor.py` (待实现)
- 实现 `FeatureExtractor` 接口的具体子类 `CLIPVisualExtractor`。
- 负责加载预训练的 CLIP 模型。
- 调用 `video_utils.py` 进行帧采样，然后将采样的图像帧送入 CLIP 模型。
- 对 CLIP 输出进行聚合 (`mean` / `max`)。
- 调用 `projection.py` 进行维度投影。
- 应用 L2-norm。
- 包含视频文件存在性、投影输出形状等断言。

### `main.py` (待实现)
- 程序入口点。
- 使用 `argparse` 处理命令行参数 (例如 `--config config.yaml`)。
- 调用 `AppConfig.from_yaml` 加载配置。
- 实例化 `CLIPVisualExtractor`。
- 调用 `run_pipeline` 协调整个特征提取流程。

### 核心数据流 (如初始计划所示)
```
main.py
  ├─ load_config() → AppConfig
  ├─ CLIPVisualExtractor(cfg)
  │     ├─ clip.load("ViT-B/32")
  │     └─ LinearProjection(512→1024)
  └─ run_pipeline(cfg, extractor)
        ├─ parse_dev_txt() → list[DialogueRecord]
        ├─ build_context_index() → {(dialogue_id, name) → [video_paths]}
        └─ For each UtteranceRecord:
              ├─ speaker (index 0): feature extraction process
              ├─ listener (index 1..N-1): feature extraction process or zero fallback
              ├─ torch.cat → [N, 1024]
              ├─ assert shape[0]==N, shape[1]==1024
              ├─ torch.save(.pt)
              └─ load-back verify
```

## 7. 当前开发进度
- [x] 初始化 Git 仓库。
- [x] 重构 `config.yaml` 结构，将抽取器配置模块化，并更新 `src/config.py` 以匹配新结构。为 `config.yaml` 和 `src/config.py` 添加详细注释。

## 8. 下一步开发计划
- 实现 `dev.txt` 解析器（`src/parser.py`），包含全面的防御性断言。
