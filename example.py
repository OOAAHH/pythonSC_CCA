#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python Seurat CCA 使用示例
==========================

此脚本展示如何使用Python版本的Seurat CCA算法来整合单细胞RNA-seq数据。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 导入我们的Seurat CCA实现
from seurat_cca import SeuratCCA


def create_simulation_data(n_genes=1000, n_cells1=500, n_cells2=700, seed=42):
    """创建模拟的单细胞数据集"""
    np.random.seed(seed)
    
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
    
    return df1, df2


def visualize_integration(original_data1, original_data2, integrated_data, filename="integration_results.png"):
    """可视化整合前后的效果"""
    # 将数据集合并用于降维可视化
    batch_labels = np.array(['Batch1'] * original_data1.shape[1] + ['Batch2'] * original_data2.shape[1])
    
    # 合并数据
    original_combined = np.hstack([original_data1.values, original_data2.values])
    integrated_combined = np.hstack([original_data1.values, integrated_data])
    
    # 使用PCA进行降维
    pca = PCA(n_components=30)
    pca_original = pca.fit_transform(original_combined.T)
    pca_integrated = pca.fit_transform(integrated_combined.T)
    
    # 使用t-SNE进行进一步降维可视化
    tsne = TSNE(n_components=2, random_state=42)
    tsne_original = tsne.fit_transform(pca_original)
    tsne_integrated = tsne.fit_transform(pca_integrated)
    
    # 绘图
    plt.figure(figsize=(14, 6))
    
    # 整合前
    plt.subplot(1, 2, 1)
    for batch in np.unique(batch_labels):
        mask = batch_labels == batch
        plt.scatter(tsne_original[mask, 0], tsne_original[mask, 1], label=batch, alpha=0.7)
    plt.title('整合前 (t-SNE)')
    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')
    plt.legend()
    
    # 整合后
    plt.subplot(1, 2, 2)
    for batch in np.unique(batch_labels):
        mask = batch_labels == batch
        plt.scatter(tsne_integrated[mask, 0], tsne_integrated[mask, 1], label=batch, alpha=0.7)
    plt.title('整合后 (t-SNE)')
    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"可视化结果已保存到 {filename}")


def main():
    """主函数：演示SeuratCCA的用法"""
    print("创建模拟数据...")
    data1, data2 = create_simulation_data(n_genes=500, n_cells1=300, n_cells2=400)
    
    print(f"数据集1形状: {data1.shape}")
    print(f"数据集2形状: {data2.shape}")
    
    # 初始化SeuratCCA
    print("\n初始化SeuratCCA...")
    cca = SeuratCCA(verbose=True)
    
    # 运行CCA
    print("\n运行CCA...")
    cca_results = cca.run_cca(data1, data2, num_cc=20)
    
    # 寻找整合锚点，传入CCA结果
    print("\n寻找整合锚点...")
    anchor_results = cca.find_integration_anchors(data1, data2, cca_results=cca_results, k_anchor=5)
    
    # 整合数据
    print("\n整合数据...")
    integrated_data = cca.integrate_data(data1, data2, anchors=anchor_results['anchors'])
    
    print(f"\n整合数据形状: {integrated_data.shape}")
    
    # 可视化结果
    print("\n可视化整合效果...")
    visualize_integration(data1, data2, integrated_data)
    
    # 多数据集整合示例
    print("\n\n创建第三个数据集用于多数据集整合示例...")
    data3, _ = create_simulation_data(n_genes=500, n_cells1=350, n_cells2=200, seed=43)
    print(f"数据集3形状: {data3.shape}")
    
    # 准备多个数据集
    datasets = [data1, data2, data3]
    
    # 方法1：参考数据集整合
    print("\n使用参考数据集整合方法...")
    integrated_datasets = cca.integrate_multiple_datasets(
        datasets, 
        k_anchor=5, 
        reference_dataset=0  # 使用第一个数据集作为参考
    )
    
    print(f"整合结果数量: {len(integrated_datasets)}")
    for i, dataset in enumerate(integrated_datasets):
        print(f"整合后数据集 {i} 形状: {dataset.shape}")
    
    # 方法2：成对整合
    print("\n使用成对整合方法...")
    integrated_dataset = cca.integrate_multiple_datasets_pairwise(
        datasets, 
        k_anchor=5
    )
    
    print(f"整合后数据集形状: {integrated_dataset.shape}")
    
    print("\n演示完成！")


if __name__ == "__main__":
    main() 