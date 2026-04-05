import subprocess, sys

# Auto-install dependencies
for pkg in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']:
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

import ast, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
SEED = 42

# ── 1. Load & Preprocess ──────────────────────────────────────────────────────
# Put tmdb_5000_movies.csv in the same folder as this script
df = pd.read_csv('tmdb_5000_movies.csv')

def get_names(s):
    try:
        return [x['name'] for x in ast.literal_eval(s)]
    except:
        return []

df['genre_list'] = df['genres'].apply(get_names)
top_genres = [g for g, _ in Counter(sum(df['genre_list'].tolist(), [])).most_common(8)]
for g in top_genres:
    df[f'g_{g}'] = df['genre_list'].apply(lambda l: int(g in l))

feat_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count'] \
            + [f'g_{g}' for g in top_genres]

X = df[feat_cols].copy()
X[['budget', 'revenue']] = X[['budget', 'revenue']].replace(0, np.nan)
X.fillna(X.median(), inplace=True)
for c in ['budget', 'revenue', 'popularity', 'vote_count']:
    X[c] = np.log1p(X[c])

X_sc  = StandardScaler().fit_transform(X)
X_pca = PCA(2, random_state=SEED).fit_transform(X_sc)
print(f'Data ready — shape: {X_sc.shape}')

# ── 2. Elbow (SSE) + Silhouette ───────────────────────────────────────────────
K = range(2, 11)
sse, sil = [], []
for k in K:
    km  = KMeans(k, random_state=SEED, n_init=10)
    lbl = km.fit_predict(X_sc)
    sse.append(km.inertia_)
    sil.append(silhouette_score(X_sc, lbl))

BEST_K = list(K)[np.argmax(sil)]
print(f'Best k = {BEST_K}  |  Silhouette = {max(sil):.3f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K, sse, 'bo-')
axes[0].set(title='Elbow — SSE', xlabel='k', ylabel='SSE')
axes[1].plot(K, sil, 'gs-')
axes[1].axvline(BEST_K, color='red', linestyle='--', label=f'Best k={BEST_K}')
axes[1].set(title='Silhouette Score', xlabel='k')
axes[1].legend()
plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=120)
plt.show()

# ── 3. K-Means ────────────────────────────────────────────────────────────────
km = KMeans(BEST_K, random_state=SEED, n_init=10)
df['km_lbl'] = km.fit_predict(X_sc)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['km_lbl'], cmap='tab10', alpha=0.5, s=10)
ax.set(title=f'K-Means (k={BEST_K}) — PCA 2D', xlabel='PC1', ylabel='PC2')
plt.colorbar(sc, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=120)
plt.show()

print('\nCluster Profiles:')
print(df.groupby('km_lbl')[['vote_average', 'popularity', 'budget', 'revenue']].mean().round(2))

# ── 4. DBSCAN ─────────────────────────────────────────────────────────────────
dists = np.sort(NearestNeighbors(n_neighbors=10).fit(X_sc).kneighbors(X_sc)[0][:, -1])[::-1]
eps_val = float(np.percentile(dists, 90))

df['db_lbl'] = DBSCAN(eps=eps_val, min_samples=10).fit_predict(X_sc)
n_cl    = len(set(df['db_lbl'])) - (1 if -1 in df['db_lbl'].values else 0)
n_noise = (df['db_lbl'] == -1).sum()
print(f'\nDBSCAN — Clusters: {n_cl}  |  Noise: {n_noise} ({n_noise / len(df) * 100:.1f}%)')

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['db_lbl'], cmap='tab10', alpha=0.4, s=10)
ax.set(title=f'DBSCAN (eps={eps_val:.2f}) — PCA 2D', xlabel='PC1', ylabel='PC2')
plt.tight_layout()
plt.savefig('dbscan_clusters.png', dpi=120)
plt.show()

# ── 5. Evaluation & Comparison ────────────────────────────────────────────────
km_sil = silhouette_score(X_sc, df['km_lbl'])
km_dbi = davies_bouldin_score(X_sc, df['km_lbl'])

mask   = df['db_lbl'] != -1
db_sil = silhouette_score(X_sc[mask], df.loc[mask, 'db_lbl'])     if df.loc[mask, 'db_lbl'].nunique() >= 2 else float('nan')
db_dbi = davies_bouldin_score(X_sc[mask], df.loc[mask, 'db_lbl']) if df.loc[mask, 'db_lbl'].nunique() >= 2 else float('nan')

results = pd.DataFrame({
    'Algorithm'        : ['K-Means', 'DBSCAN'],
    'n_clusters'       : [BEST_K, n_cl],
    'Noise pts'        : [0, n_noise],
    'Silhouette'       : [round(km_sil, 4), round(db_sil, 4)],
    'Davies-Bouldin'   : [round(km_dbi, 4), round(db_dbi, 4)],
})
print('\n===== Evaluation Results =====')
print(results.to_string(index=False))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
a1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['km_lbl'], cmap='tab10', alpha=0.4, s=8)
a1.set_title(f'K-Means k={BEST_K}  |  Sil={km_sil:.3f}')
a2.scatter(X_pca[:, 0], X_pca[:, 1], c=df['db_lbl'], cmap='tab10', alpha=0.4, s=8)
a2.set_title(f'DBSCAN eps={eps_val:.2f}  |  Sil={db_sil:.3f}')
plt.suptitle('K-Means vs DBSCAN', fontsize=13)
plt.tight_layout()
plt.savefig('comparison.png', dpi=120)
plt.show()

print("""
Summary:
  K-Means -> needs k upfront, spherical clusters, fast, great for segmentation
  DBSCAN  -> no k needed, handles noise & outliers, finds arbitrary shapes
""")