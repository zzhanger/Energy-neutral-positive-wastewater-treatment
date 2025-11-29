# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 15:45:38 2025

@author: zz1405
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from adjustText import adjust_text


province_abbr = {
    'Beijing': 'BJ', 'Tianjin': 'TJ', 'Hebei': 'HEB', 'Shanxi': 'SX', 
    'Inner Mongolia': 'IM', 'Liaoning': 'LN', 'Jilin': 'JL', 
    'Heilongjiang': 'HLJ', 'Shanghai': 'SH', 'Jiangsu': 'JS', 
    'Zhejiang': 'ZJ', 'Anhui': 'AH', 'Fujian': 'FJ', 'Jiangxi': 'JX', 
    'Shandong': 'SD', 'Henan': 'HEN', 'Hubei': 'HUB', 'Hunan': 'HUN', 
    'Guangdong': 'GD', 'Guangxi': 'GX', 'Hainan': 'HAN', 'Chongqing': 'CQ', 
    'Sichuan': 'SC', 'Guizhou': 'GZ', 'Yunnan': 'YN', 'Xizang': 'XZ', 
    'Shaanxi': 'SNX', 'Gansu': 'GS', 'Qinghai': 'QH', 'Ningxia': 'NX', 
    'Xinjiang': 'XJ'
}

def load_and_preprocess_data():
    df = pd.read_csv('Data for figure 3b.csv')
    df.columns = [col.replace('m^3', 'm³') for col in df.columns]
    

    numeric_cols = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'TPinf', 'SSinf', 'Average_flow__MGD_',
                   'Flow based EUI (kWh/m³)', 'COD removal based EUI (kWh/kg COD removed)', 'TPeff', 'CODeff']
    

    df['Province_abbr'] = df['Province'].map(province_abbr)
    
    # Convert to numeric and drop missing values
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols + ['Province_abbr'])
    
    return df, numeric_cols

def calculate_province_averages(df, numeric_cols):

    return df.groupby('Province_abbr')[numeric_cols].mean().reset_index()

def perform_pca_analysis(province_avg):
    """Perform PCA analysis for both EUI types"""
    scaler = StandardScaler()
    
    # PCA for Flow based EUI
    flow_eui_pca_cols = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'Average_flow__MGD_']
    flow_eui_pca_cols = list(set(flow_eui_pca_cols))
    
    flow_eui_scaled = scaler.fit_transform(province_avg[flow_eui_pca_cols])
    pca_flow = PCA(n_components=2)
    flow_eui_pca = pca_flow.fit_transform(flow_eui_scaled)
    province_avg[['PC1_flow', 'PC2_flow']] = flow_eui_pca
    
    # PCA for COD removal based EUI
    cod_eui_pca_cols = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'SSinf', 'TPinf', 'Average_flow__MGD_']
    cod_eui_pca_cols = list(set(cod_eui_pca_cols))
    
    cod_eui_scaled = scaler.fit_transform(province_avg[cod_eui_pca_cols])
    pca_cod = PCA(n_components=2)
    cod_eui_pca = pca_cod.fit_transform(cod_eui_scaled)
    province_avg[['PC1_cod', 'PC2_cod']] = cod_eui_pca
    
    return province_avg, pca_flow, pca_cod, flow_eui_pca_cols, cod_eui_pca_cols

def perform_clustering(province_avg, flow_eui_pca, cod_eui_pca):
    """Perform clustering analysis"""
    n_clusters = min(6, len(province_avg)-1)
    
    # Flow EUI clustering
    kmeans_flow = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    province_avg['Cluster_flow'] = kmeans_flow.fit_predict(flow_eui_pca)
    
    # COD EUI clustering
    kmeans_cod = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    province_avg['Cluster_cod'] = kmeans_cod.fit_predict(cod_eui_pca)
    
    return province_avg

def safe_get_representative(df, cluster, cluster_col='Cluster_flow'):
    """Safely get representative province for a cluster (province closest to median features)"""
    cluster_data = df[df[cluster_col] == cluster]
    if len(cluster_data) == 0:
        return None
    
    # Select appropriate feature columns based on cluster type
    if cluster_col == 'Cluster_flow':
        feature_cols = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'Average_flow__MGD_']
    else:
        feature_cols = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'SSinf', 'TPinf', 'Average_flow__MGD_']

    median_values = cluster_data[feature_cols].median()
    distances = cluster_data[feature_cols].sub(median_values).abs().sum(axis=1)

    return cluster_data.loc[distances.idxmin()]

def plot_pca_clusters(ax, data, pca, color_col, title, pca_type='flow'):
    """Universal function for plotting PCA clusters"""
    # Select appropriate coordinates and cluster columns based on PCA type
    if pca_type == 'flow':
        x_col, y_col, cluster_col = 'PC1_flow', 'PC2_flow', 'Cluster_flow'
        pca_cols = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'Average_flow__MGD_']
    else:
        x_col, y_col, cluster_col = 'PC1_cod', 'PC2_cod', 'Cluster_cod'
        pca_cols = ['CODinf', 'BODinf', 'TNinf', 'NH3_Ninf', 'SSinf', 'TPinf', 'Average_flow__MGD_']
    
    # Plot cluster boundaries
    cluster_colors = plt.cm.tab20.colors
    n_clusters_actual = data[cluster_col].nunique()
    
    for cluster in range(n_clusters_actual):
        cluster_data = data[data[cluster_col] == cluster]
        points = cluster_data[[x_col, y_col]].values
        
        if len(points) > 2:
            try:
                hull = ConvexHull(points)
                ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 
                       color=cluster_colors[cluster], linestyle='--', alpha=0.4)
                ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                       color=cluster_colors[cluster], alpha=0.1)
            except:
                # Skip if convex hull cannot be calculated
                pass


    scatter = ax.scatter(
        x=data[x_col],
        y=data[y_col],
        c=data[color_col],
        cmap='viridis',
        s=150,
        alpha=0.9,
        edgecolors='w',
        linewidth=1
    )


    if 'm³' in color_col:
        cbar_label = color_col.replace('m³', r'$m^3$')  # Use LaTeX format superscript
    else:
        cbar_label = color_col
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label(cbar_label, fontsize=24)
    
    # Use province abbreviation labels
    texts = []
    for _, row in data.iterrows():
        texts.append(ax.text(
            row[x_col], row[y_col],
            row['Province_abbr'],
            fontsize=16,
            ha='center',
            va='center',
            alpha=0.8,
            fontweight='bold'
        ))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, alpha=0.3))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=24)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=24)
    ax.set_title(title, fontsize=24, pad=15)
    ax.grid(alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    print(f"\n{title} - PCA Loading Matrix:")
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=pca_cols
    )
    print(feature_importance.round(3))

def print_analysis_results(province_avg, pca_flow, pca_cod, flow_eui_pca_cols, cod_eui_pca_cols):
    """Print analysis results"""
    print("\n" + "="*60)
    print("PCA Explained Variance Ratios")
    print("="*60)
    print(f"Flow EUI PCA - PC1: {pca_flow.explained_variance_ratio_[0]:.3f}, PC2: {pca_flow.explained_variance_ratio_[1]:.3f}")
    print(f"COD EUI PCA - PC1: {pca_cod.explained_variance_ratio_[0]:.3f}, PC2: {pca_cod.explained_variance_ratio_[1]:.3f}")

    print("\n" + "="*60)
    print("Variables Used in PCA Analysis")
    print("="*60)
    print("Flow based EUI PCA variables:", flow_eui_pca_cols)
    print("COD removal based EUI PCA variables:", cod_eui_pca_cols)

    print("\n" + "="*60)
    print("Clustering Analysis Results")
    print("="*60)

    # Flow EUI clustering information
    print("\nFlow EUI Cluster Distribution:")
    for cluster in sorted(province_avg['Cluster_flow'].unique()):
        cluster_data = province_avg[province_avg['Cluster_flow'] == cluster]
        provinces = cluster_data['Province_abbr'].tolist()
        print(f"Cluster {cluster}: {provinces}")

    # COD EUI clustering information
    print("\nCOD EUI Cluster Distribution:")
    for cluster in sorted(province_avg['Cluster_cod'].unique()):
        cluster_data = province_avg[province_avg['Cluster_cod'] == cluster]
        provinces = cluster_data['Province_abbr'].tolist()
        print(f"Cluster {cluster}: {provinces}")

def main():
    df, numeric_cols = load_and_preprocess_data()

    province_avg = calculate_province_averages(df, numeric_cols)

    province_avg, pca_flow, pca_cod, flow_eui_pca_cols, cod_eui_pca_cols = perform_pca_analysis(province_avg)

    province_avg = perform_clustering(province_avg, 
                                    province_avg[['PC1_flow', 'PC2_flow']].values,
                                    province_avg[['PC1_cod', 'PC2_cod']].values)

    plt.figure(figsize=(20, 10))


    ax1 = plt.subplot(121)
    plot_pca_clusters(ax1, province_avg, pca_flow, 'Flow based EUI (kWh/m³)', 
                     'PCA: Flow based EUI with Flow and TP Variables', 'flow')

    ax2 = plt.subplot(122)
    plot_pca_clusters(ax2, province_avg, pca_cod, 'COD removal based EUI (kWh/kg COD removed)', 
                     'PCA: COD EUI with TP and COD Variables', 'cod')

    plt.tight_layout()
    plt.show()
    
    print_analysis_results(province_avg, pca_flow, pca_cod, flow_eui_pca_cols, cod_eui_pca_cols)

if __name__ == "__main__":
    main()