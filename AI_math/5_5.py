#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment 
@File    ：5_5.py
@IDE     ：PyCharm 
@Author  ：SYZ
@Date    ：2023/12/14 10:01
5. 附件（Samples.mat）给出了 1000 个样本在某条有可能致病的染色体片段上的 9445 个
位点的编码信息（Samples 矩阵大小为 1000×9445），其中前 500 个样本为健康者，后 500
个样本为患者。采用如方差分析、卡方检验等方法，绘制不同位点与患病显著性关系的散点
图，给出与该种疾病相关性最大的前 10 个位点。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import chi2_contingency
import pandas as pd

# Your code for loading the .mat file and analysis goes here


# The correct file has been uploaded. Let's attempt to load it again and proceed with the analysis.
mat = loadmat('Samples.mat')


# Assuming the samples matrix is stored under the key 'Samples' and has a size of 1000x9445
# We'll check for the correct key in the .mat file, as it might be different
keys = [key for key in mat.keys() if not key.startswith('__')]
print(keys)
if keys:
    # Assuming the first key that does not start with '__' is the one we're interested in
    samples_key = keys[0]
    samples = mat[samples_key]

    # Split the data into healthy (first 500 samples) and patients (last 500 samples)
    healthy = samples[:500, :]
    patients = samples[500:, :]

    # Conduct chi-square tests for each locus
    p_values = []
    for locus in range(samples.shape[1]):
        # Construct the contingency table for the current locus
        table = np.array([np.bincount(healthy[:, locus]), np.bincount(patients[:, locus])])
        if table.shape[1] == 2: # Ensure there are two groups to compare
            chi2, p, dof, ex = chi2_contingency(table, correction=False)
            p_values.append(p)
        else:
            # If there's only one group, we assign a p-value of 1
            # indicating no difference between healthy and patient groups
            p_values.append(1)

    # Convert the p-values to a numpy array
    p_values = np.array(p_values)

    # Find the loci with the 10 smallest p-values
    top_10_loci = np.argsort(p_values)[:10]
    top_10_p_values = np.sort(p_values)[:10]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(samples.shape[1]), -np.log10(p_values), alpha=0.5, label='All loci')
    plt.scatter(top_10_loci, -np.log10(top_10_p_values), color='red', label='Top 10 loci')
    plt.xlabel('Locus Index')
    plt.ylabel('-log10(p-value)')
    plt.title('Locus Significance in Disease Association')
    plt.axhline(-np.log10(0.05), color='grey', linestyle='dashed', linewidth=1, label='Significance threshold')
    plt.legend()
    plt.show()

    # Output the loci and p-values
    top_10_results = pd.DataFrame({
        'Locus': top_10_loci,
        'p-value': top_10_p_values
    })

    # Save the results to a CSV file
    csv_filename = 'top_10_loci.csv'
    top_10_results.to_csv(csv_filename, index=False)

    top_10_results, csv_filename
else:
    raise ValueError("No appropriate key found in the .mat file for the samples matrix.")

