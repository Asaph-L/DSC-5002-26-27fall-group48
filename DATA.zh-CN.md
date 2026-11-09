# 数据来源与准备

[English](DATA.md)

本项目使用 Austin Reese 在 Kaggle 发布的二手车数据集。项目指定使用 2021 年 5 月 6 日发布的第 10 版，其中包含刊登日期，与提案所述约 426,880 条记录、26 个字段相符。新版数据的记录范围可能不同。`price` 字段记录卖家以美元标出的挂牌价，是否成交仍需另行核实。[Kaggle 数据集](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) · [第 10 版](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/versions/10)

## 下载文件

打开上方的第 10 版链接并下载数据。若 Kaggle 提示登录，请使用自己的账号。将压缩包中的 `vehicles.csv` 解压至仓库根目录下的 `data/raw/vehicles.csv`。运行 notebook 前请保留文件的原始字节，避免用电子表格软件打开后重新保存。

也可以在仓库根目录运行以下命令，通过官方客户端获取同一文件。[KaggleHub 下载说明](https://github.com/Kaggle/kagglehub#download-dataset)

```bash
python -m pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('austinreese/craigslist-carstrucks-data/versions/10', path='vehicles.csv', output_dir='data/raw')"
```

## 验证数据版本

第 10 版原始压缩包中的 `vehicles.csv` 包含 426,880 条刊登记录和 26 个字段，文件大小为 1,447,955,215 字节。解压后 CSV 的 SHA-256 校验值为 `54c60e20aa494a73f7f9170bff1e38b4800472b767cab8f9dab7da5dcc261a5c`。

运行以下命令，确认输出与上述校验值一致。

```bash
python -c "import hashlib; f=open('data/raw/vehicles.csv','rb'); print(hashlib.file_digest(f, 'sha256').hexdigest())"
```

清洗前先核对表头和行数。分享分析结果时，请同时注明数据版本和文件校验值。[01 · 导入与数据概况](notebooks/01_import_and_profile.ipynb)

## Group 48 清洗数据包

压缩包为 4.74 MB，其中的 `listings.csv` 是 Notebook 02 生成的原始清洗表，共 393,985 条记录、18 列。将文件解压到 `data/processed/listings.csv`，即可运行 Notebook 03 和 04。压缩包附有中英文说明和校验值。需要读取 Kaggle 原始快照的 notebooks 仍需准备 `data/raw/vehicles.csv`。[下载清洗数据 ZIP](downloads/DSC5002_Group48_cleaned_listings.zip)

## GitHub 协作范围

Kaggle 将此数据集标记为 **CC0（公共领域）**。原始刊登记录可能包含卖家填写的描述、车辆标识、精确位置和链接。Craigslist 的使用条款限制转载刊登内容，且原始 CSV 超过 GitHub 常规的 100 MiB 单文件上限。[Craigslist 使用条款](https://www.craigslist.org/about/terms.of.use/en) · [GitHub 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

组员各自从 Kaggle 获取同一版原始数据，并保存在 Git 忽略的本地 `data/` 目录中。

上方获准公开的清洗 ZIP 是本项目唯一公开的逐条记录快照，包含挂牌价、车型年份、里程和州别等结构化分析字段。压缩包已排除刊登 ID、链接、描述、车辆识别号码（VIN）和地理坐标。其他本地生成的数据和缓存保存在 Git 忽略的目录中。公开的 notebook 输出使用汇总图表。
