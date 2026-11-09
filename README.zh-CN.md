# Craigslist 二手车挂牌价分析

[English](README.md)

这是 SDSC 5002 的小组项目。我们用探索性数据分析研究二手车挂牌价如何随车龄、里程表读数、车辆属性和所在州而变化。分析以卖家在 Craigslist 上的报价为对象，结论限于样本中观察到的关联。

[项目提案](docs/proposal.zh-CN.md) · [数据来源与准备](DATA.zh-CN.md)

## 运行 notebooks

请使用 Python 3.11 或更新版本。在仓库根目录创建并激活虚拟环境，然后安装依赖。

```bash
python -m venv .venv
# macOS/Linux：source .venv/bin/activate
# Windows PowerShell：.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

按[数据准备说明](DATA.zh-CN.md)下载指定版本，将 `vehicles.csv` 放在 `data/raw/`。运行 `python -m jupyterlab` 启动 Jupyter，然后按顺序运行下表中的 notebooks。

| Notebook | 内容 |
| --- | --- |
| [01 · 数据导入与概况](notebooks/01_import_and_profile.ipynb) | 检查文件、字段、缺失值和重复挂牌 ID，汇总数值字段。 |
| [02 · 清洗与派生变量](notebooks/02_clean_and_derive.ipynb) | 核对清洗前后的样本量与价格，计算车龄，并在本地生成 `data/processed/listings.csv`。 |
| [03 · 挂牌价、车龄与里程](notebooks/03_price_age_mileage.ipynb) | 绘制价格分布和分箱密度图，检查极端价格，比较不同筛选条件下的相关系数。 |
| [04 · 按车辆类别和州比较](notebooks/04_categories_and_states.ipynb) | 比较车况、燃料、车型和品牌的价格分布，按州展示中位数及样本量。 |
| [05 · 控制变量后的关联与价格预测](notebooks/05_adjusted_associations_and_prediction.ipynb) | 对挂牌价取对数后做回归，并用较晚发布的挂牌比较中位数基线、岭回归和梯度提升树。 |

分析保留正挂牌价，并给特别低或高的价格加复核标记。每项分析使用该问题所需字段齐全的记录，并报告相应的样本量。具体筛选规则及其影响见清洗与分析 notebooks。

## 小组协作

用 GitHub Issues 记录任务和待解决的问题。每次修改开一个分支，通过 Pull Request 简述改动、重新运行了哪些 notebooks，以及筛选条件或结果是否有变化。修改同一个 notebook 前先在组内协调。把共用的数据处理逻辑放在 `src/`，在 notebooks 中呈现分析。提交前检查输出，只保留汇总表和图；数据文件、原始刊登内容、标识符和凭据留在本地。修改分析口径时，请同步更新中英文版本。
