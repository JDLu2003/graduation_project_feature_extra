# 中文 reference 覆盖评估

这个目录用于评估 `reference_face_zh` 对中文数据集的理论覆盖面，重点回答两个问题：

1. 当前脸库已经覆盖了多少高频角色？
2. 在不同的 reference 图片数门槛下，能覆盖多少角色、speaker 语句和 listener 语句？

## 技术路线

当前方案不直接跑人脸模型，而是先评估“参考脸库是否足够支撑开放集检索方案”。

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

## 默认路径

- 数据集标注文件：`/data16T_1/sunshengzhe/lujiading/data_zh/dev/dev.txt`
- reference 根目录：`/data16T_2/sunshengzhe/reference_face_zh`
- 输出目录：`scripts/zh_reference_coverage/artifacts/latest`

## 主要文件

- `run_reference_coverage_eval.py`
  主评估脚本。
- `Makefile`
  提供快捷命令。
- `artifacts/`
  存放中间结果和最终报告。

## 输出内容

运行后会生成：

- `summary.json`
- `reference_inventory.json`
- `suspicious_aliases.json`
- `role_reference_coverage.csv`
- `threshold_coverage.csv`
- `report.md`

其中 `report.md` 是最适合直接阅读的中文报告。

## 使用方式

在仓库根目录：

```bash
make zh-ref-coverage
```

或进入当前目录：

```bash
make run
```

## 结果解读建议

1. 优先关注 `threshold_coverage.csv` 中 `min_ref_images=10` 或 `15` 的覆盖率。
2. 优先补齐 `report.md` 中“高频但缺失 reference 的角色”。
3. 对“已覆盖但样本仍偏少”的高频角色补充多样化图片，而不是只补数量。
4. 对疑似别名做人工确认，例如简称/本名/外号混用的问题。
