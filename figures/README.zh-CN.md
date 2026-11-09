# 图表索引

[English](README.md)

这里收录初步探索、建模和四项深入研究的 31 张正式图表，以及一张初版海报的公开预览。每张图都可下载 PNG，提供矢量版的图也列出了 SVG 或 PDF。

图表基于 Kaggle v10 的 Craigslist 数据，金额均为美元挂牌价。各图的样本口径和方法详见对应 notebook。

## 初步探索

[Notebook 03](../notebooks/03_price_age_mileage.ipynb) · [Notebook 04](../notebooks/04_categories_and_states.ipynb)

| 图表 | 下载 |
| --- | --- |
| 主要价格区间与完整价格范围 | [PNG](exploratory/01_price_ranges.png) |
| 车龄和里程的挂牌价密度 | [PNG](exploratory/02_age_mileage_density.png) |
| 不同车况的挂牌价 | [PNG](exploratory/03_price_by_condition.png) |
| 不同燃料类型的挂牌价 | [PNG](exploratory/04_price_by_fuel.png) |
| 不同车型的挂牌价 | [PNG](exploratory/05_price_by_vehicle_type.png) |
| 不同制造商的挂牌价 | [PNG](exploratory/06_price_by_manufacturer.png) |
| 各州挂牌价中位数 | [PNG](exploratory/07_state_median_prices.png) |

## 初步模型比较

[Notebook 05](../notebooks/05_adjusted_associations_and_prediction.ipynb)

| 图表 | 下载 |
| --- | --- |
| 较晚时间留出集的预测误差 | [PNG](model-baseline/01_holdout_model_comparison.png) |
| 调整后的车龄和里程对比 | [PNG](model-baseline/02_adjusted_associations.png) |

## Q1 · 价格分布与异常值

[Notebook 06](../notebooks/06_price_distribution_and_anomalies.ipynb)

| 图表 | 下载 |
| --- | --- |
| 完整与主要价格分布 | [PNG](q1/01_price_distribution.png) · [PDF](q1/01_price_distribution.pdf) |
| 价格尾部概率 | [PNG](q1/02_tail_probabilities.png) · [PDF](q1/02_tail_probabilities.pdf) |
| 常见价格上的聚集 | [PNG](q1/03_price_heaping.png) · [PDF](q1/03_price_heaping.pdf) |
| 不同车型的筛除比例 | [PNG](q1/04_vehicle_type_screening.png) · [PDF](q1/04_vehicle_type_screening.pdf) |
| 同类车辆异常值筛查 | [PNG](q1/05_peer_relative_screen.png) · [PDF](q1/05_peer_relative_screen.pdf) |

## Q2 · 车龄与里程

[Notebook 07](../notebooks/07_age_mileage_nonlinearity.ipynb)

| 图表 | 下载 |
| --- | --- |
| 车龄和里程的原始关系 | [PNG](q2/01_raw_age_mileage.png) · [SVG](q2/01_raw_age_mileage.svg) |
| 老车可比样本对照 | [PNG](q2/02_older_age_overlap.png) · [SVG](q2/02_older_age_overlap.svg) |
| 调整后的车龄和里程关系 | [PNG](q2/03_adjusted_profiles.png) · [SVG](q2/03_adjusted_profiles.svg) |
| 车龄和里程的交互 | [PNG](q2/04_age_mileage_interaction.png) · [SVG](q2/04_age_mileage_interaction.svg) |
| 条件分位数预测 | [PNG](q2/05_quantile_predictions.png) · [SVG](q2/05_quantile_predictions.svg) |

## Q3 · 车辆类别差异

[Notebook 08](../notebooks/08_comparable_vehicle_groups.ipynb)

| 图表 | 下载 |
| --- | --- |
| 各车辆类别的价格分布 | [PNG](q3/01_group_distributions.png) · [SVG](q3/01_group_distributions.svg) |
| 原始价差与可比样本加权价差 | [PNG](q3/02_overlap_weighted_contrasts.png) · [SVG](q3/02_overlap_weighted_contrasts.svg) |
| 各车况的价格分布 | [PNG](q3/03_condition_distributions.png) · [SVG](q3/03_condition_distributions.svg) |
| 不同车龄段的车况价差 | [PNG](q3/04_condition_gap_by_age.png) · [SVG](q3/04_condition_gap_by_age.svg) |
| 各车辆类别的车况缺失情况 | [PNG](q3/05_condition_missingness.png) · [SVG](q3/05_condition_missingness.svg) |

## Q4 · 州别价格差异

[Notebook 09](../notebooks/09_state_price_differences.ipynb)

| 图表 | 下载 |
| --- | --- |
| 州别原始中位价与模型调整后价差 | [PNG](q4/state_price_maps.png) · [PDF](q4/state_price_maps.pdf) |
| 调整前后的州别排名 | [PNG](q4/state_rank_changes.png) · [PDF](q4/state_rank_changes.pdf) |
| 州别价格差异及区间 | [PNG](q4/state_intervals.png) · [PDF](q4/state_intervals.pdf) |
| 不同模型和时间段的敏感性检验 | [PNG](q4/state_gap_sensitivity.png) · [PDF](q4/state_gap_sensitivity.pdf) |

## 时间滚动验证

[Notebook 10](../notebooks/10_rolling_validation_and_error_analysis.ipynb)

| 图表 | 下载 |
| --- | --- |
| 各时间验证窗口的模型误差 | [PNG](validation/rolling_validation_mae.png) · [PDF](validation/rolling_validation_mae.pdf) |
| 不同挂牌价与车龄段的模型误差 | [PNG](validation/price_and_age_error_profile.png) · [PDF](validation/price_and_age_error_profile.pdf) |
| 移除特征组后的模型误差 | [PNG](validation/feature_ablation.png) · [PDF](validation/feature_ablation.pdf) |
