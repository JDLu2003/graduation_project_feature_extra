# 中文 reference 覆盖评估

这个目录现在包含两类评估：

1. 静态覆盖率评估：当前脸库已经覆盖了多少高频角色。
2. 流水线评估：`MTCNN -> 预训练 FaceNet -> 聚类 -> reference 检索` 这条方案在中文数据集上能覆盖多少 utterance / speaker / participant。

## 技术路线

这里有两条技术路线。

### 1. 静态覆盖率评估

不直接跑人脸模型，而是先评估“参考脸库是否足够支撑开放集检索方案”。

核心思路：

1. 解析中文数据集的 `dev.txt`。
2. 统计每个角色的出现频次，包含：
   - `dialogue_count`
   - `utterance_count`
   - `speaker_count`
   - `listener_count`
3. 扫描 `/data16T_2/sunshengzhe/reference_face_zh`，统计每个 identity 目录中的有效图片数量。
4. 用 `reference_image_count >= threshold` 近似表示“这个角色可进入 gallery 检索库”，再计算覆盖率。
5. 输出缺失高频角色、样本偏少的高频角色、空目录和疑似别名，作为后续清洗和补库依据。

### 2. FaceNet 聚类检索流水线

这条流水线会直接跑视频，步骤如下：

1. 使用 `MTCNN` 对采样帧做人脸检测，不区分说话人和听话人。
2. 使用预训练 `FaceNet` (`InceptionResnetV1`, `vggface2`) 批量提取 512 维 embedding。
3. 使用余弦阈值做在线聚类，得到若干 `cluster`。
4. 用 cluster centroid 与 `reference_face_zh` 的 prototype 做 1:N 检索匹配。
5. 通过 `top1 score` 与 `top1-top2 margin` 做拒识。
6. 最后统计：
   - utterance 有脸覆盖率
   - utterance 已知角色命中率
   - speaker 命中率
   - participant 命中率
   - listener 槽位命中率
   - 角色覆盖率

## 默认路径

- 数据集标注文件：`/data16T_1/sunshengzhe/lujiading/data_zh/dev/dev.txt`
- 视频目录：`/data16T_1/sunshengzhe/lujiading/data_zh/dev/Video_dev`
- reference 根目录：`/data16T_2/sunshengzhe/reference_face_zh`
- 输出目录：`scripts/zh_reference_coverage/artifacts/latest`

## 主要文件

- `run_reference_coverage_eval.py`
  静态覆盖率评估脚本。
- `run_facenet_cluster_pipeline.py`
  MTCNN + FaceNet + 聚类 + reference 检索评估脚本。
- `Makefile`
  提供快捷命令。
- `artifacts/`
  存放中间结果和最终报告。

## 输出内容

静态覆盖率脚本会生成：

- `summary.json`
- `reference_inventory.json`
- `suspicious_aliases.json`
- `role_reference_coverage.csv`
- `threshold_coverage.csv`
- `report.md`

其中 `report.md` 是最适合直接阅读的中文报告。

流水线脚本会额外生成：

- `pipeline_summary.json`
- `pipeline_report.md`
- `face_records.csv`
- `extracted_faces/`
- `query_embeddings.npy`
- `cluster_centroids.npy`
- `cluster_matches.csv`
- `cluster_gallery/`
- `utterance_predictions.csv`
- `reference_bank_summary.json`
- `run_config.json`

其中：

1. `extracted_faces/` 会保存 MTCNN 检出的所有人脸 crop。
2. `cluster_gallery/` 会把聚类后的样本按 cluster 复制到新的子目录中。
3. 每个 cluster 子目录下都会生成 `说明.txt`，写明：
   - 最终命名结果
   - top 候选 reference
   - cluster 大小
   - 与 reference 的匹配分数
   - cluster 内部一致性统计
   - 判定时使用的关键超参数

## 使用方式

在仓库根目录：

```bash
make zh-ref-coverage
```

或进入当前目录：

```bash
make run
```

如果要跑完整的人脸聚类检索流水线：

```bash
make pipeline
```

先做小规模冒烟：

```bash
make pipeline-smoke
```

## A6000 10G 推荐起始参数

如果你希望显存控制在 10G 以内，建议先从这组参数开始：

```bash
make pipeline \
  DEVICE=cuda \
  NUM_FRAMES=6 \
  DETECT_BATCH_SIZE=8 \
  EMBED_BATCH_SIZE=256 \
  REFERENCE_EMBED_BATCH_SIZE=256 \
  FACE_VERIFY_THRESHOLD=0.75 \
  MAX_FACES_PER_FRAME=8 \
  USE_FP16=1
```

调参建议：

1. 优先增大 `EMBED_BATCH_SIZE`，它最影响 GPU 吞吐。
2. 如果检测阶段显存吃紧或波动大，先减小 `DETECT_BATCH_SIZE`。
3. 如果同一角色被拆得太碎，降低 `CLUSTER_THRESHOLD`。
4. 如果误识别多，提高 `REFERENCE_MATCH_THRESHOLD` 或 `REFERENCE_MATCH_MARGIN`。

## FaceNet 一致性相关超参数

流水线里和“判断两张脸是否像同一个人”最相关的超参数有 4 个：

1. `CLUSTER_THRESHOLD`
   控制新的人脸 embedding 是否并入已有 cluster。
   可以理解为“聚类层面的同人阈值”。

2. `REFERENCE_MATCH_THRESHOLD`
   控制 cluster centroid 与 reference prototype 至少要相似到什么程度，才允许命名。

3. `REFERENCE_MATCH_MARGIN`
   控制 top1 和 top2 候选之间至少要拉开多少差距，避免强行误认。

4. `FACE_VERIFY_THRESHOLD`
   用于 cluster 内部一致性分析。
   脚本会统计每个 cluster 里有多少张脸与 centroid 的相似度超过这个阈值，并写进 `说明.txt`。

一个保守起点可以是：

```bash
CLUSTER_THRESHOLD=0.72
REFERENCE_MATCH_THRESHOLD=0.80
REFERENCE_MATCH_MARGIN=0.03
FACE_VERIFY_THRESHOLD=0.75
```

## 结果解读建议

1. 优先关注 `threshold_coverage.csv` 中 `min_ref_images=10` 或 `15` 的覆盖率。
2. 优先补齐 `report.md` 中“高频但缺失 reference 的角色”。
3. 对“已覆盖但样本仍偏少”的高频角色补充多样化图片，而不是只补数量。
4. 对疑似别名做人工确认，例如简称/本名/外号混用的问题。
