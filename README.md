# Python-SeuratCCA

Python实现的Seurat CCA (典型相关分析) 算法，用于单细胞数据整合。该实现解决了R版本中的整数限制问题，可以处理大规模单细胞数据集的整合任务。

## 介绍

Seurat是一个流行的R包，用于单细胞数据分析。其中，CCA（Canonical Correlation Analysis，典型相关分析）是Seurat用于数据整合的重要方法之一。然而，由于R语言的整数限制，原版Seurat在处理大规模数据集时可能会遇到性能问题。

本实现使用Python重写了Seurat的CCA算法，解决了R语言的整数限制问题，同时保持了与原始算法相同的功能，包括：

1. 运行CCA寻找共同变异
2. 查找整合锚点
3. 基于锚点整合数据
4. 支持多数据集整合

## 数据整合概念说明

在使用CCA进行单细胞数据整合时，有一些重要概念需要明确：

1. **整合的实际过程**：在CCA整合中，第一个数据集（通常是参考数据集）保持不变，而其他数据集会被转换到与参考数据集相同的空间中。整合的结果不是所有数据的混合物，而是参考数据集和其他被转换的数据集的集合。

2. **整合数据的结构**：
   - 使用`integrate_data`函数时，返回的只是转换后的第二个数据集（不包括参考数据集）
   - 使用`integrate_multiple_datasets`函数时，返回列表包含参考数据集（未变）和所有转换后的其他数据集
   - 使用`integrate_multiple_datasets_pairwise`函数时，返回的是所有数据集按顺序整合后的结果

3. **可视化整合效果**：为了正确评估整合效果，应将原始参考数据集和转换后的数据集合并后一起可视化。这样才能观察到不同批次的细胞是否混合均匀，从而判断整合是否成功。

## 安装

### 依赖项

- Python 3.6+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Annoy

### 安装依赖项

```bash
pip install numpy pandas scipy scikit-learn annoy
```

## 使用方法

### 基本用法

```python
from seurat_cca import SeuratCCA
import pandas as pd

# 假设有两个单细胞数据集：data1和data2
# 数据格式为pandas DataFrame，行为基因，列为细胞

# 初始化SeuratCCA对象
cca = SeuratCCA(verbose=True)

# 运行CCA
cca_results = cca.run_cca(data1, data2, num_cc=20)

# 寻找整合锚点
anchor_results = cca.find_integration_anchors(data1, data2, k_anchor=5)

# 整合数据
integrated_data = cca.integrate_data(data1, data2, anchors=anchor_results['anchors'])

# 注意：integrated_data仅包含转换后的data2
# 完整的整合数据集应该包括data1和integrated_data
complete_integrated_data = pd.concat([data1, pd.DataFrame(integrated_data, 
                                                        index=data1.index, 
                                                        columns=data2.columns)], axis=1)
```

### 多数据集整合

```python
# 假设有多个单细胞数据集
datasets = [data1, data2, data3]

# 方法1：将所有数据集与参考数据集整合
integrated_datasets = cca.integrate_multiple_datasets(
    datasets, 
    k_anchor=5, 
    reference_dataset=0  # 使用第一个数据集作为参考
)
# 注意：integrated_datasets[0]是未变的参考数据集
# integrated_datasets[1]和integrated_datasets[2]是转换后的data2和data3

# 方法2：成对整合（按序列顺序）
integrated_dataset = cca.integrate_multiple_datasets_pairwise(
    datasets, 
    k_anchor=5
)
# 这里integrated_dataset包含了所有数据集整合后的结果
```

### 参数说明

#### run_cca

- `object1`, `object2`: 要整合的两个数据集
- `standardize`: 是否标准化矩阵 (默认: True)
- `num_cc`: 要计算的典型相关向量数量 (默认: 20)
- `seed`: 随机种子 (默认: 42)

#### find_integration_anchors

- `object1`, `object2`: 要整合的两个数据集
- `features`: 用于查找锚点的特征 (默认: None，使用所有特征)
- `k_filter`: 用于过滤锚点的邻居数量 (默认: 200)
- `k_anchor`: 查找锚点时使用的邻居数量 (默认: 5)
- `dims`: 用于整合的维度 (默认: range(0, 10))

#### integrate_data

- `object1`, `object2`: 要整合的两个数据集
- `anchors`: 整合锚点 (默认: None，会自动计算)
- `features_to_integrate`: 要整合的特征 (默认: None，使用所有特征)
- `k_weight`: 计算锚点权重时考虑的邻居数量 (默认: 100)
- `sd_weight`: 控制高斯核带宽的权重 (默认: 1)

## 示例

代码包含了两个示例函数：

1. `simple_example()`: 展示如何整合两个模拟数据集
2. `multiple_dataset_example()`: 展示如何整合三个模拟数据集

运行示例：

```python
from seurat_cca import simple_example, multiple_dataset_example

# 运行两个数据集的整合示例
results = simple_example()

# 运行多个数据集的整合示例
multi_results = multiple_dataset_example()
```

## 与AnnData/Scanpy兼容

如果你正在使用Scanpy生态系统，可以通过以下方式将AnnData对象转换为可用于SeuratCCA的格式：

```python
import scanpy as sc
import pandas as pd

# 加载AnnData对象
adata1 = sc.read_h5ad('dataset1.h5ad')
adata2 = sc.read_h5ad('dataset2.h5ad')

# 转换为DataFrame（基因 x 细胞）
data1 = pd.DataFrame(adata1.X.T, index=adata1.var_names, columns=adata1.obs_names)
data2 = pd.DataFrame(adata2.X.T, index=adata2.var_names, columns=adata2.obs_names)

# 接下来可以使用SeuratCCA
cca = SeuratCCA()
integrated_data = cca.integrate_data(data1, data2)

# 将整合结果转回AnnData
integrated_adata = sc.AnnData(X=integrated_data.T)
integrated_adata.obs_names = data2.columns
integrated_adata.var_names = data2.index
```

## 注意事项

- 该实现旨在保持与原始Seurat CCA算法功能相同，但在处理大数据方面具有更好的性能
- 对于非常大的数据集，建议使用稀疏矩阵格式
- CCA计算可能耗费内存，请确保有足够的RAM

## 致谢

这个实现参考了Seurat包的CCA算法。感谢Seurat开发团队开发了这个优秀的工具。
