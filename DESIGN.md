# 设计说明文档：多模态对话特征提取

## Context

当前项目已完成 VisualCLIP 特征提取策略的核心实现，本文档旨在清晰描述：
1. `FeatureExtractor` 抽象接口的设计意图与各方法语义
2. `VisualCLIPStrategy` 的完整设计，包括 `CLIPEncoder`、`context_index`、说话人/非说话人处理流程

---

## 文档结构

### 1. Extractor 接口设计（`src/extractors/base.py`）

`FeatureExtractor` 抽象接口定义了特征提取策略应遵循的核心行为。其设计遵循开闭原则 (OCP)，允许通过实现新策略来扩展功能，而不必修改核心 pipeline 代码。

| 成员 | 签名 | 说明 |
|------|------|------|
| `prepare` | `(dialogue_records, video_base_dir) -> None` | 准备阶段，用于执行一次性或预计算任务，如构建上下文索引。输入为解析后的对话记录和视频根目录。 |
| `extract_speaker` | `(video_path: Path) -> Tensor[1, DIMS]` | 从指定的视频路径中提取说话人的特征向量。返回一个 `[1, DIMS]` 维度的 Tensor。 |
| `extract_non_speaker` | `(person_name: str, dialogue_id: int) -> Tensor[1, DIMS]` | 提取非说话人的特征向量。该方法允许根据人名和对话ID回溯上下文信息，并定义回退策略（如返回零向量）。返回一个 `[1, DIMS]` 维度的 Tensor。 |
| `output_dim` | `@property -> int` | 返回该特征提取器输出的特征向量维度。 |

---

### 2. VisualCLIP 策略设计

`VisualCLIPStrategy` 是 `FeatureExtractor` 接口的一个具体实现，专注于使用 CLIP 模型从视频中提取视觉特征。

#### 2.1 组成模块

- `CLIPEncoder`：封装了 CLIP 模型的推理逻辑。它负责处理视频帧，通过 CLIP 模型编码，并进行后续的维度投影和归一化，最终输出符合目标维度要求的特征向量。
- `context_index`：一个在 `prepare` 阶段构建的查找表。它存储了每个说话人在特定对话中对应的视频路径，主要用于非说话人特征提取时回溯其作为说话人时的视觉信息。
- `VisualCLIPStrategy`：作为核心策略类，它组合了 `CLIPEncoder` 和 `context_index`，并实现了 `FeatureExtractor` 接口中定义的 `prepare`、`extract_speaker` 和 `extract_non_speaker` 方法。

#### 2.2 CLIPEncoder 工作流程

`CLIPEncoder` 负责将视频文件转换为 L2 归一化的目标维度特征向量，其内部工作流程如下：

```
视频文件 (Path)
  ↓ 1. sample_frames(video_path, strategy: str)：根据预设策略（如 uniform/middle/first）从视频中采样图像帧。
  ↓ 2. 图像预处理：将采样的帧从 BGR 格式转换为 RGB，并应用 CLIP 模型所需的预处理步骤（如 resize, normalize）。
  ↓ 3. 堆叠帧：将处理后的所有帧堆叠成一个 PyTorch Tensor，形状为 `[num_frames, 3, H, W]`。
  ↓ 4. CLIP 图像编码：使用预训练的 CLIP 模型（处于 `no_grad` 模式）编码图像，得到 `[num_frames, clip_output_dim]` 的特征。
  ↓ 5. 特征聚合：对多帧特征进行聚合操作（如求平均 `mean` 或取最大 `max`），将特征维度降为 `[1, clip_output_dim]`。
  ↓ 6. 线性投影与归一化：将聚合后的特征通过一个线性层 `LinearProjection` 投影到目标维度 `target_dim`，并进行 L2 归一化 `F.normalize`，最终输出 `[1, target_dim]` 的特征向量。
```

**特殊处理**：
- 如果视频文件不存在，`sample_frames` 阶段将触发 `assert` 报错。
- 如果采样帧结果为空，`CLIPEncoder` 将返回一个全零向量 `[1, target_dim]`，并打印警告。

#### 2.3 context_index 数据结构与构建

`context_index` 用于存储对话中每个人作为说话人时的视频片段信息，以便在提取非说话人特征时进行查找。

**数据结构示例**:
```python
Dict[Tuple[int, str], List[Path]]
# 示例：{(18, "Monica"): [Path("Video_en_dev/C_18/C_18_U_3.mp4"), Path("Video_en_dev/C_18/C_18_U_7.mp4")], ...}
```

**构建逻辑**：
在 `VisualCLIPStrategy` 的 `prepare` 方法中完成构建。它会遍历所有解析出的对话 utterance 记录。对于每一条记录，如果某人是说话人，其对应的视频路径（**仅当文件真实存在于磁盘时**）会被添加到 `context_index` 中，键为 `(dialogue_id, person_name)`，值为该人在该对话中作为说话人时的所有视频路径列表。

#### 2.4 说话人 vs 非说话人提取策略

`VisualCLIPStrategy` 针对说话人和非说话人采用不同的特征提取策略：

| 角色 | 方法 | 策略 |
|------|------|------|
| 说话人 | `extract_speaker(video_path)` | 直接调用 `CLIPEncoder` 对当前 utterance 对应的 `video_path` 进行编码，获取特征。 |
| 非说话人 | `extract_non_speaker(person_name, dialogue_id)` | 1. 在 `context_index` 中查找 `(dialogue_id, person_name)` 对应的视频路径列表。2. 如果找到视频，对所有这些视频调用 `CLIPEncoder` 提取特征，然后将所有特征进行平均，并进行 L2 归一化。3. 如果 `context_index` 中没有找到该非说话人作为说话人时的上下文视频（即无上下文），则执行回退策略。 |
| 非说话人（无上下文） | fallback | 返回一个维度为 `[1, target_dim]` 的全零 PyTorch Tensor。 |

#### 2.5 最终 Tensor 组装

在 pipeline 流程中，对于每个 utterance，最终的特征 Tensor 组装严格按照 `dev.txt` 中人物出现的顺序进行：

```
[speaker_feat]           → 说话人特征，始终位于 Tensor 的索引 0
[listener_1_feat]        → 第一个听众特征，位于索引 1
...
[listener_N_feat]        → 第 N 个听众特征，位于索引 N
──────────────────────────
torch.cat → 拼接成最终的 `[N+1, 1024]` 维度的 Tensor
```
这个最终的 Tensor 将被保存为 `.pt` 文件。

---

## 关键文件路径

| 文件 | 作用 |
|------|------|
| `src/extractors/base.py` | 定义 `FeatureExtractor` 抽象基类 |
| `src/extractors/visual_clip/clip_encoder.py` | `CLIPEncoder` 的具体实现，封装 CLIP 模型推理逻辑 |
| `src/extractors/visual_clip/strategy.py` | `VisualCLIPStrategy` 的核心实现，组合 `CLIPEncoder` 和 `context_index` |
| `src/extractors/visual_clip/context_index.py` | 负责构建和管理 `context_index` 数据结构 |
| `src/config.py` | 定义了项目的各项配置，包括特征维度、视频采样策略等 |
| `src/pipeline.py` | 实现了整个特征提取的调用流程，协调各模块工作 |
