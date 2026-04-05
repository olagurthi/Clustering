# 🎬 TMDB Movies Clustering Analysis

A machine learning project that applies unsupervised clustering algorithms to the TMDB 5000 Movies dataset to discover natural groupings among films based on budget, revenue, popularity, ratings, and genre.

---

## 📁 Project Structure

```
project/
│
├── tmdb_clustering.py        # Main script
├── tmdb_5000_movies.csv      # Dataset
├── elbow_silhouette.png      # Elbow & silhouette plots
├── kmeans_clusters.png       # K-Means cluster visualization
├── knn_distance.png          # k-NN distance plot for DBSCAN eps
├── dbscan_clusters.png       # DBSCAN cluster visualization
└── comparison.png            # Side-by-side comparison plot
```

---

## 📊 Dataset

**TMDB 5000 Movies** — 4803 movies with features including:
- Budget & Revenue
- Popularity & Vote Average & Vote Count
- Runtime
- Genres (parsed from JSON)

---

## ⚙️ Methods Used

| Step | Method |
|------|--------|
| Preprocessing | Log transform, median imputation, StandardScaler |
| Dimensionality Reduction | PCA (2D for visualization) |
| Clustering 1 | K-Means |
| Clustering 2 | DBSCAN |
| Optimal k selection | Elbow Method (SSE) + Silhouette Score |
| Evaluation | Silhouette Score, Davies-Bouldin Index |

---

## 🔧 How to Run

**Requirements:** Python 3.8+, PyCharm (or any IDE)

1. Clone or download this repository
2. Place `tmdb_5000_movies.csv` in the same folder as `tmdb_clustering.py`
3. Open in PyCharm and click **Run** — dependencies install automatically

Or via terminal:
```bash
python tmdb_clustering.py
```

---

## 📈 Results

| Algorithm | Clusters | Noise Points | Silhouette ↑ | Davies-Bouldin ↓ |
|-----------|----------|-------------|--------------|-----------------|
| K-Means   | 10       | 0           | 0.1546       | 1.6055          |
| DBSCAN    | 11       | 213 (4.4%)  | 0.1401       | 1.6110          |

---

## 🔍 Personal Observation

Running both algorithms on this dataset revealed that K-Means with k=10 performed better overall, as the silhouette score steadily improved with each additional cluster rather than peaking early, suggesting movies naturally spread across many fine-grained groups — from low-budget indie films to big-budget blockbusters. DBSCAN, while slightly weaker on metrics, offered something K-Means could not: it flagged 213 movies (4.4%) as outliers that genuinely don't belong to any cluster, such as films with unusual genre combinations or mismatched budget-to-revenue ratios. In this case, K-Means is better for clean segmentation and interpretation, while DBSCAN is more honest about the noise in the data.

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```
> All installed automatically when running the script.

---

## 👤 Author
Ola Gurthi
