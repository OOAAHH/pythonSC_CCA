import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
import annoy
import itertools

class SeuratCCA:
    """
    Python实现的Seurat CCA (Canonical Correlation Analysis)算法，
    用于单细胞数据整合，解决R版本的整数限制问题
    """
    
    def __init__(self, verbose=True):
        """
        初始化SeuratCCA对象
        
        参数:
            verbose: 是否显示详细信息
        """
        self.verbose = verbose
        self.cca_embeddings = None
        self.singular_values = None
        self.features = None
    
    def standardize(self, mat):
        """
        标准化矩阵 - 将列缩放为单位方差和均值0
        
        参数:
            mat: 输入矩阵 (特征 x 细胞)
            
        返回:
            标准化后的矩阵
        """
        if sp.issparse(mat):
            mat = mat.toarray()
        
        scaler = StandardScaler(with_mean=True, with_std=True)
        return scaler.fit_transform(mat.T).T
    
    def run_cca(self, object1, object2, standardize=True, num_cc=20, seed=42):
        """
        在两个数据集之间执行CCA
        
        参数:
            object1: 第一个数据矩阵 (特征 x 细胞1)
            object2: 第二个数据矩阵 (特征 x 细胞2)
            standardize: 是否标准化矩阵
            num_cc: 要计算的典型相关向量数量
            seed: 随机种子
            
        返回:
            包含CCA结果的字典
        """
        if seed is not None:
            np.random.seed(seed)
        
        cells1 = object1.columns if hasattr(object1, 'columns') else np.arange(object1.shape[1])
        cells2 = object2.columns if hasattr(object2, 'columns') else np.arange(object2.shape[1])
        
        # 转换为numpy数组
        if sp.issparse(object1):
            object1 = object1.toarray()
        elif isinstance(object1, pd.DataFrame):
            object1 = object1.values
            
        if sp.issparse(object2):
            object2 = object2.toarray()
        elif isinstance(object2, pd.DataFrame):
            object2 = object2.values
            
        # 标准化
        if standardize:
            if self.verbose:
                print("标准化数据...")
            object1 = self.standardize(object1)
            object2 = self.standardize(object2)
        
        # 计算交叉乘积矩阵
        if self.verbose:
            print("计算交叉乘积矩阵...")
        mat3 = np.dot(object1.T, object2)
        
        # 使用SVD计算
        if self.verbose:
            print(f"计算前{num_cc}个奇异值...")
        u, d, vt = svds(mat3, k=num_cc)
        
        # 对奇异值和向量进行排序（从大到小）
        idx = np.argsort(d)[::-1]
        d = d[idx]
        u = u[:, idx]
        vt = vt[idx, :]
        
        # 获取v（vt的转置）
        v = vt.T
        
        # 构建CCA数据 - 按行堆叠u和v
        cca_data = np.vstack((u, v))
        
        # 确保所有向量的符号一致性
        for i in range(cca_data.shape[1]):
            if cca_data[0, i] < 0:
                cca_data[:, i] *= -1
        
        # 存储结果
        self.cca_embeddings = cca_data
        self.singular_values = d
        
        # 构建结果
        result = {
            'ccv': cca_data,
            'd': d
        }
        
        return result
    
    def find_integration_anchors(self, object1, object2, features=None, cca_results=None,
                           k_filter=200, k_anchor=5, k_score=30, 
                           max_features=200, nn_method='annoy',
                           n_trees=50, dims=range(0, 10)):
        """
        在两个数据集之间寻找整合锚点
        
        参数:
            object1: 第一个数据对象
            object2: 第二个数据对象
            features: 用于查找锚点的特征
            cca_results: 已有的CCA结果
            k_filter: 用于过滤锚点的邻居数量
            k_anchor: 选择锚点时要使用的邻居数量
            k_score: 评分锚点时要使用的邻居数量
            max_features: 用于锚点评分的最大特征数量
            nn_method: 用于查找邻居的方法 ('annoy', 'exact')
            n_trees: Annoy索引中的树数量
            dims: 用于整合的维度
        
        返回:
            包含锚点信息的字典
        """
        if cca_results is not None:
            self.cca_embeddings = cca_results['ccv']
            self.singular_values = cca_results['d']
        elif self.cca_embeddings is None:
            if self.verbose:
                print("首先运行CCA...")
            self.run_cca(object1, object2)
        
        # 分割CCA嵌入
        n_cells1 = object1.shape[1] if hasattr(object1, 'shape') else len(object1.columns)
        
        # 现在嵌入的每行是一个细胞，每列是一个CCA成分
        cca_cells1 = self.cca_embeddings[:n_cells1, :]  # 第一个数据集的细胞
        cca_cells2 = self.cca_embeddings[n_cells1:, :]  # 第二个数据集的细胞
        
        # 选择要使用的维度
        cca_cells1_dims = cca_cells1[:, dims]  # 选择特定的CCA成分
        cca_cells2_dims = cca_cells2[:, dims]
        
        # 查找互相邻居
        if self.verbose:
            print(f"查找锚点候选项，使用k_anchor={k_anchor}...")
            
        if nn_method == 'annoy':
            # 使用Annoy查找近似最近邻
            cells1_to_cells2 = self._find_nn_annoy(cca_cells1_dims, cca_cells2_dims, k_anchor, n_trees)
            cells2_to_cells1 = self._find_nn_annoy(cca_cells2_dims, cca_cells1_dims, k_anchor, n_trees)
        else:
            # 使用精确方法查找邻居
            cells1_to_cells2 = self._find_nn_exact(cca_cells1_dims, cca_cells2_dims, k_anchor)
            cells2_to_cells1 = self._find_nn_exact(cca_cells2_dims, cca_cells1_dims, k_anchor)
            
        # 找到互相邻居
        anchors = []
        
        for i in range(n_cells1):
            neighbors_ab = cells1_to_cells2[i]
            
            for pos, j in enumerate(neighbors_ab):
                # 检查是否为互相邻居
                j_neighbors = cells2_to_cells1[j]
                if i in j_neighbors:
                    # 存储锚点和评分
                    score = 1.0 - (pos / k_anchor)  # 简单评分方法
                    anchors.append((i, j, score))
        
        if self.verbose:
            print(f"找到{len(anchors)}个锚点")
            
        # 过滤锚点（如果需要）
        if k_filter < len(anchors) and k_filter > 0:
            if self.verbose:
                print(f"按分数过滤到前{k_filter}个锚点...")
            anchors.sort(key=lambda x: x[2], reverse=True)
            anchors = anchors[:k_filter]
        
        # 转换为numpy数组，确保使用整数类型
        anchor_array = np.array(anchors, dtype=int)
        
        return {
            'anchors': anchor_array,
            'cells1_to_cells2': cells1_to_cells2,
            'cells2_to_cells1': cells2_to_cells1
        }
    
    def _find_nn_annoy(self, query_data, index_data, k, n_trees=50):
        """使用Annoy查找近似最近邻"""
        n_dims = query_data.shape[1]
        index = annoy.AnnoyIndex(n_dims, 'euclidean')
        
        # 添加数据点到索引
        for i in range(index_data.shape[0]):
            index.add_item(i, index_data[i])
            
        # 构建索引
        index.build(n_trees)
        
        # 查询
        results = []
        for i in range(query_data.shape[0]):
            neighbors = index.get_nns_by_vector(query_data[i], k)
            results.append(neighbors)
            
        return results
    
    def _find_nn_exact(self, query_data, index_data, k):
        """使用精确方法查找最近邻"""
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(index_data)
        distances, indices = nbrs.kneighbors(query_data)
        return indices.tolist()
    
    def integrate_data(self, object1, object2, anchors=None, features_to_integrate=None, 
                       weight_reduction=None, k_weight=100, sd_weight=1):
        """
        基于锚点整合两个数据集
        
        参数:
            object1: 第一个数据对象
            object2: 第二个数据对象
            anchors: 整合锚点（如果为None，将自动寻找）
            features_to_integrate: 要整合的特征
            weight_reduction: 用于计算锚点权重的降维
            k_weight: 计算锚点权重时要考虑的邻居数量
            sd_weight: 控制高斯核带宽的权重
            
        返回:
            整合后的数据矩阵
        """
        if anchors is None:
            if self.verbose:
                print("未提供锚点，寻找整合锚点...")
            anchor_result = self.find_integration_anchors(object1, object2)
            anchors = anchor_result['anchors']
        
        # 如果未指定特征，使用所有共同特征
        if features_to_integrate is None:
            if hasattr(object1, 'index') and hasattr(object2, 'index'):
                features_to_integrate = list(set(object1.index).intersection(set(object2.index)))
            else:
                # 假设两个对象具有相同的特征
                features_to_integrate = range(object1.shape[0])
        
        # 提取数据
        if isinstance(object1, pd.DataFrame):
            data1 = object1.loc[features_to_integrate].values
        else:
            data1 = object1[features_to_integrate, :]
            
        if isinstance(object2, pd.DataFrame):
            data2 = object2.loc[features_to_integrate].values
        else:
            data2 = object2[features_to_integrate, :]
        
        # 计算整合向量
        if self.verbose:
            print("计算整合向量...")
            
        integration_vectors = []
        for cell1, cell2, score in anchors:
            # 确保使用整数索引
            cell1_idx = int(cell1)
            cell2_idx = int(cell2)
            
            # 获取锚点细胞的数据
            vec1 = data1[:, cell1_idx]
            vec2 = data2[:, cell2_idx]
            
            # 计算差异向量
            diff_vec = vec2 - vec1
            integration_vectors.append((diff_vec, score))
        
        # 计算每个细胞的权重（默认使用CCA空间中的距离）
        if self.verbose:
            print("计算每个细胞的锚点权重...")
            
        if weight_reduction is None and self.cca_embeddings is not None:
            weight_reduction = self.cca_embeddings
        
        n_cells1 = object1.shape[1] if hasattr(object1, 'shape') else len(object1.columns)
        n_cells2 = object2.shape[1] if hasattr(object2, 'shape') else len(object2.columns)
        
        # 使用CCA空间计算权重
        weights = self._calculate_weights(anchors, n_cells1, n_cells2, 
                                        weight_reduction, k_weight, sd_weight)
        
        # 整合数据
        if self.verbose:
            print("整合数据...")
            
        integrated_data = np.zeros((len(features_to_integrate), n_cells2))
        
        for i in range(n_cells2):
            correction = np.zeros(len(features_to_integrate))
            weight_sum = 0
            
            for a, (diff_vec, score) in enumerate(integration_vectors):
                w = weights[i, a]
                if w > 0:
                    correction += w * diff_vec
                    weight_sum += w
            
            if weight_sum > 0:
                correction /= weight_sum
                
            integrated_data[:, i] = data2[:, i] - correction
            
        if self.verbose:
            print("数据整合完成")
            
        return integrated_data
    
    def _calculate_weights(self, anchors, n_cells1, n_cells2, 
                        reduction_data, k_weight=100, sd_weight=1):
        """计算每个细胞与每个锚点的权重"""
        n_anchors = len(anchors)
        weights = np.zeros((n_cells2, n_anchors))
        
        # 提取锚点坐标 - 确保使用整数索引
        anchor_coords = np.array([reduction_data[int(anchor[0]), :] for anchor in anchors])
        
        # 获取第二个数据集的坐标
        cells2_coords = reduction_data[n_cells1:n_cells1+n_cells2, :]
        
        # 为每个细胞计算与锚点的距离并应用高斯核
        for i in range(n_cells2):
            cell_coord = cells2_coords[i]
            
            # 计算与所有锚点的欧氏距离
            distances = np.sqrt(np.sum((anchor_coords - cell_coord)**2, axis=1))
            
            # 应用高斯核转换距离为权重
            sigma = np.std(distances) * sd_weight
            if sigma == 0:
                sigma = 1e-10  # 避免除以零
                
            weights[i] = np.exp(-distances**2 / (2 * sigma**2))
            
            # 仅保留前k_weight个最大权重
            if k_weight < n_anchors:
                threshold = np.sort(weights[i])[-k_weight]
                weights[i] = np.where(weights[i] >= threshold, weights[i], 0)
                
        return weights

    def integrate_multiple_datasets(self, datasets, features=None, k_anchor=5, k_filter=200, 
                                    k_weight=100, sd_weight=1, dims=range(0, 10), 
                                    reference_dataset=None, num_cc=20, seed=42):
        """
        整合多个数据集
        
        参数:
            datasets: 要整合的数据集列表
            features: 用于整合的特征（如果为None，将使用所有共同特征）
            k_anchor: 查找锚点时使用的邻居数量
            k_filter: 用于过滤锚点的邻居数量
            k_weight: 计算锚点权重时考虑的邻居数量
            sd_weight: 控制高斯核带宽的权重
            dims: 用于整合的维度
            reference_dataset: 参考数据集索引（如果为None，将使用第一个数据集作为参考）
            num_cc: 每次CCA计算的典型相关向量数量
            seed: 随机种子
            
        返回:
            整合后的数据列表
        """
        if len(datasets) < 2:
            raise ValueError("至少需要两个数据集来执行整合")
            
        # 确定参考数据集
        if reference_dataset is None:
            reference_dataset = 0
            
        # 如果未指定特征，找出所有数据集共有的特征
        if features is None:
            if all(hasattr(dataset, 'index') for dataset in datasets):
                common_features = set(datasets[0].index)
                for dataset in datasets[1:]:
                    common_features = common_features.intersection(set(dataset.index))
                features = list(common_features)
                if self.verbose:
                    print(f"使用{len(features)}个共同特征进行整合")
            else:
                # 假设所有数据集具有相同的特征
                features = range(datasets[0].shape[0])
                
        # 存储整合结果
        integrated_datasets = []
        
        # 首先将参考数据集添加到结果中（不需要整合）
        reference_data = datasets[reference_dataset]
        if isinstance(reference_data, pd.DataFrame):
            integrated_datasets.append(reference_data.copy())
        else:
            integrated_datasets.append(reference_data.copy() if hasattr(reference_data, 'copy') else reference_data)
            
        # 整合策略：所有数据集都与参考数据集配对
        for i, dataset in enumerate(datasets):
            if i == reference_dataset:
                continue  # 跳过参考数据集
                
            if self.verbose:
                print(f"\n整合数据集 {i} 与参考数据集 {reference_dataset}")
                
            # 运行CCA
            self.run_cca(reference_data, dataset, num_cc=num_cc, seed=seed)
            
            # 查找锚点
            anchor_results = self.find_integration_anchors(
                reference_data, dataset, 
                features=features, 
                k_anchor=k_anchor, 
                k_filter=k_filter,
                dims=dims
            )
            
            # 整合数据
            integrated_data = self.integrate_data(
                reference_data, dataset,
                anchors=anchor_results['anchors'],
                features_to_integrate=features,
                k_weight=k_weight,
                sd_weight=sd_weight
            )
            
            # 转换回与原始数据相同的格式
            if isinstance(dataset, pd.DataFrame):
                # 创建DataFrame
                if isinstance(features, list) and all(isinstance(f, str) for f in features):
                    # 特征是字符串
                    integrated_df = pd.DataFrame(integrated_data, index=features, columns=dataset.columns)
                else:
                    # 特征是索引
                    if hasattr(dataset, 'index'):
                        integrated_df = pd.DataFrame(integrated_data, index=dataset.index, columns=dataset.columns)
                    else:
                        integrated_df = pd.DataFrame(integrated_data, columns=dataset.columns)
                        
                integrated_datasets.append(integrated_df)
            else:
                integrated_datasets.append(integrated_data)
                
        return integrated_datasets
    
    def integrate_multiple_datasets_pairwise(self, datasets, features=None, k_anchor=5, k_filter=200, 
                                            k_weight=100, sd_weight=1, dims=range(0, 10), 
                                            num_cc=20, seed=42, sample_tree=None):
        """
        使用成对整合策略整合多个数据集
        
        参数:
            datasets: 要整合的数据集列表
            features: 用于整合的特征（如果为None，将使用所有共同特征）
            k_anchor: 查找锚点时使用的邻居数量
            k_filter: 用于过滤锚点的邻居数量
            k_weight: 计算锚点权重时考虑的邻居数量
            sd_weight: 控制高斯核带宽的权重
            dims: 用于整合的维度
            num_cc: 每次CCA计算的典型相关向量数量
            seed: 随机种子
            sample_tree: 整合顺序的数据结构，如果为None，将自动计算
            
        返回:
            整合后的数据集
        """
        if len(datasets) < 2:
            raise ValueError("至少需要两个数据集来执行整合")
            
        # 如果未指定特征，找出所有数据集共有的特征
        if features is None:
            if all(hasattr(dataset, 'index') for dataset in datasets):
                common_features = set(datasets[0].index)
                for dataset in datasets[1:]:
                    common_features = common_features.intersection(set(dataset.index))
                features = list(common_features)
                if self.verbose:
                    print(f"使用{len(features)}个共同特征进行整合")
            else:
                # 假设所有数据集具有相同的特征
                features = range(datasets[0].shape[0])
                
        # 确定整合顺序
        if sample_tree is None:
            # 默认策略：先整合前两个数据集，然后整合结果与第三个数据集，依此类推
            integration_steps = []
            current = 0
            
            for i in range(1, len(datasets)):
                integration_steps.append((current, i))
                current = -i  # 表示前一步整合的结果
                
            if self.verbose:
                print(f"使用默认整合顺序: {integration_steps}")
        else:
            integration_steps = sample_tree
            
        # 存储中间结果（索引<0表示整合结果）
        intermediate_results = list(datasets)  # 复制原始数据集
        
        # 执行整合步骤
        for step, (idx1, idx2) in enumerate(integration_steps):
            if self.verbose:
                print(f"\n整合步骤 {step+1}: 整合数据集 {idx1} 与 {idx2}")
                
            # 获取要整合的数据集
            dataset1 = intermediate_results[idx1]
            dataset2 = intermediate_results[idx2]
            
            # 运行CCA
            self.run_cca(dataset1, dataset2, num_cc=num_cc, seed=seed)
            
            # 查找锚点
            anchor_results = self.find_integration_anchors(
                dataset1, dataset2, 
                features=features, 
                k_anchor=k_anchor, 
                k_filter=k_filter,
                dims=dims
            )
            
            # 整合数据集
            integrated_data1 = dataset1  # 第一个数据集不变
            integrated_data2 = self.integrate_data(
                dataset1, dataset2,
                anchors=anchor_results['anchors'],
                features_to_integrate=features,
                k_weight=k_weight,
                sd_weight=sd_weight
            )
            
            # 转换回与原始数据相同的格式
            if isinstance(dataset2, pd.DataFrame):
                # 创建DataFrame
                if isinstance(features, list) and all(isinstance(f, str) for f in features):
                    # 特征是字符串
                    integrated_df2 = pd.DataFrame(integrated_data2, index=features, columns=dataset2.columns)
                else:
                    # 特征是索引
                    if hasattr(dataset2, 'index'):
                        integrated_df2 = pd.DataFrame(integrated_data2, index=dataset2.index, columns=dataset2.columns)
                    else:
                        integrated_df2 = pd.DataFrame(integrated_data2, columns=dataset2.columns)
                        
                integrated_data2 = integrated_df2
                
            # 合并数据集
            if isinstance(dataset1, pd.DataFrame) and isinstance(integrated_data2, pd.DataFrame):
                # 合并DataFrame
                merged_dataset = pd.concat([dataset1, integrated_data2], axis=1)
            else:
                # 合并numpy数组
                if isinstance(dataset1, np.ndarray) and isinstance(integrated_data2, np.ndarray):
                    merged_dataset = np.hstack([dataset1, integrated_data2])
                else:
                    raise ValueError("无法合并不同类型的数据集")
                    
            # 将结果存储在中间结果中
            intermediate_results.append(merged_dataset)
            
        # 返回最终整合结果
        return intermediate_results[-1]

# 使用示例1
def multiple_dataset_example():
    """使用多个数据集演示SeuratCCA"""
    # 创建三个模拟数据集
    np.random.seed(42)
    n_genes = 1000
    n_cells1 = 500
    n_cells2 = 700
    n_cells3 = 600
    
    # 创建共享和批次特定的基因表达模式
    shared_pattern = np.random.normal(0, 1, (n_genes, 5))
    batch1_specific = np.random.normal(0, 0.5, (n_genes, 3))
    batch2_specific = np.random.normal(0, 0.5, (n_genes, 3))
    batch3_specific = np.random.normal(0, 0.5, (n_genes, 3))
    
    # 生成细胞因子
    cell_factors1 = np.random.normal(0, 1, (n_cells1, 5))
    cell_factors2 = np.random.normal(0, 1, (n_cells2, 5))
    cell_factors3 = np.random.normal(0, 1, (n_cells3, 5))
    
    batch1_cell_factors = np.random.normal(0, 1, (n_cells1, 3))
    batch2_cell_factors = np.random.normal(0, 1, (n_cells2, 3))
    batch3_cell_factors = np.random.normal(0, 1, (n_cells3, 3))
    
    # 生成数据矩阵
    data1 = np.dot(shared_pattern, cell_factors1.T) + np.dot(batch1_specific, batch1_cell_factors.T)
    data1 += np.random.normal(0, 0.1, data1.shape)  # 加噪声
    
    data2 = np.dot(shared_pattern, cell_factors2.T) + np.dot(batch2_specific, batch2_cell_factors.T)
    data2 += np.random.normal(0, 0.1, data2.shape)  # 加噪声
    
    data3 = np.dot(shared_pattern, cell_factors3.T) + np.dot(batch3_specific, batch3_cell_factors.T)
    data3 += np.random.normal(0, 0.1, data3.shape)  # 加噪声
    
    # 将数据转换为 pandas DataFrame
    genes = [f"gene_{i}" for i in range(n_genes)]
    cells1 = [f"cell1_{i}" for i in range(n_cells1)]
    cells2 = [f"cell2_{i}" for i in range(n_cells2)]
    cells3 = [f"cell3_{i}" for i in range(n_cells3)]
    
    df1 = pd.DataFrame(data1, index=genes, columns=cells1)
    df2 = pd.DataFrame(data2, index=genes, columns=cells2)
    df3 = pd.DataFrame(data3, index=genes, columns=cells3)
    
    datasets = [df1, df2, df3]
    
    print(f"数据集1形状: {df1.shape}")
    print(f"数据集2形状: {df2.shape}")
    print(f"数据集3形状: {df3.shape}")
    
    # 运行SeuratCCA
    cca = SeuratCCA(verbose=True)
    
    # 方法1：将所有数据集与参考数据集整合
    print("\n方法1：将所有数据集与参考数据集整合")
    integrated_datasets = cca.integrate_multiple_datasets(
        datasets, 
        k_anchor=5, 
        reference_dataset=0
    )
    
    print(f"整合结果数量: {len(integrated_datasets)}")
    for i, dataset in enumerate(integrated_datasets):
        print(f"整合后数据集 {i} 形状: {dataset.shape}")
    
    # 方法2：成对整合
    print("\n方法2：成对整合")
    integrated_dataset = cca.integrate_multiple_datasets_pairwise(
        datasets, 
        k_anchor=5
    )
    
    print(f"整合后数据集形状: {integrated_dataset.shape}")
    
    return {
        'original_datasets': datasets,
        'integrated_datasets_method1': integrated_datasets,
        'integrated_dataset_method2': integrated_dataset
    }

# 使用示例2
def simple_example():
    """使用简单示例演示SeuratCCA"""
    # 创建两个模拟数据集
    np.random.seed(42)
    n_genes = 1000
    n_cells1 = 500
    n_cells2 = 700
    
    # 创建共享和批次特定的基因表达模式
    shared_pattern = np.random.normal(0, 1, (n_genes, 5))
    batch1_specific = np.random.normal(0, 0.5, (n_genes, 3))
    batch2_specific = np.random.normal(0, 0.5, (n_genes, 3))
    
    # 生成细胞因子
    cell_factors1 = np.random.normal(0, 1, (n_cells1, 5))
    cell_factors2 = np.random.normal(0, 1, (n_cells2, 5))
    
    batch1_cell_factors = np.random.normal(0, 1, (n_cells1, 3))
    batch2_cell_factors = np.random.normal(0, 1, (n_cells2, 3))
    
    # 生成数据矩阵
    data1 = np.dot(shared_pattern, cell_factors1.T) + np.dot(batch1_specific, batch1_cell_factors.T)
    data1 += np.random.normal(0, 0.1, data1.shape)  # 加噪声
    
    data2 = np.dot(shared_pattern, cell_factors2.T) + np.dot(batch2_specific, batch2_cell_factors.T)
    data2 += np.random.normal(0, 0.1, data2.shape)  # 加噪声
    
    # 将数据转换为 pandas DataFrame
    genes = [f"gene_{i}" for i in range(n_genes)]
    cells1 = [f"cell1_{i}" for i in range(n_cells1)]
    cells2 = [f"cell2_{i}" for i in range(n_cells2)]
    
    df1 = pd.DataFrame(data1, index=genes, columns=cells1)
    df2 = pd.DataFrame(data2, index=genes, columns=cells2)
    
    print(f"数据集1形状: {df1.shape}")
    print(f"数据集2形状: {df2.shape}")
    
    # 运行SeuratCCA
    cca = SeuratCCA(verbose=True)
    
    # 运行CCA
    print("\n运行CCA...")
    cca_results = cca.run_cca(df1, df2, num_cc=20)
    
    # 寻找整合锚点
    print("\n寻找整合锚点...")
    anchor_results = cca.find_integration_anchors(df1, df2, k_anchor=5)
    
    # 整合数据
    print("\n整合数据...")
    integrated_data = cca.integrate_data(df1, df2, anchors=anchor_results['anchors'])
    
    print(f"\n整合数据形状: {integrated_data.shape}")
    
    return {
        'data1': df1,
        'data2': df2,
        'cca_results': cca_results,
        'anchor_results': anchor_results,
        'integrated_data': integrated_data
    }

if __name__ == "__main__":
    # simple_example()
    multiple_dataset_example() 