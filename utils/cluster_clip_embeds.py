import os, numpy as np, torch
from sklearn.cluster import SpectralClustering
import umap
import matplotlib.pyplot as plt

# SETTINGS
FOLDER = "data/clip"

embeddings, file_paths = [], []
for f in sorted(os.listdir(FOLDER)):
    if f.endswith(".pt"):
        v = torch.load(os.path.join(FOLDER, f))
        embeddings.append(v.flatten().numpy())
        file_paths.append(f)
embeddings = np.stack(embeddings)

MIN_SIZE = max(3, int(0.05 * len(embeddings)))
cluster_labels = np.zeros(len(embeddings), dtype=int)
cluster_id = 1

def recurse(indices, label):
    global cluster_id
    if len(indices) <= MIN_SIZE:
        cluster_labels[indices] = label
        return
    try:
        subset = embeddings[indices]
        model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                   n_neighbors=5, assign_labels='discretize', random_state=0)
        pred = model.fit_predict(subset)
        for i in [0, 1]:
            sub_idx = indices[pred == i]
            if len(sub_idx) < MIN_SIZE:
                cluster_labels[sub_idx] = label
            else:
                recurse(sub_idx, cluster_id)
                cluster_id += 1
    except Exception:
        cluster_labels[indices] = label

recurse(np.arange(len(embeddings)), cluster_id)

# UMAP for 2D visualization
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=0)
embeddings_2d = reducer.fit_transform(embeddings)

plt.figure(figsize=(12, 10))
unique_labels = np.unique(cluster_labels)
colors = plt.cm.get_cmap('tab20', len(unique_labels))

for i, lbl in enumerate(unique_labels):
    idx = cluster_labels == lbl
    x, y = embeddings_2d[idx, 0], embeddings_2d[idx, 1]
    plt.scatter(x, y, s=10, color=colors(i), alpha=0.6)
    plt.text(np.mean(x), np.mean(y), str(lbl), fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='black'))

plt.title("Recursive Spectral Clustering (UMAP Projection)")
plt.savefig("umap_spectral_clustered.png")

np.save("cluster_labels.npy", cluster_labels)
print("Number of clusters:", len(unique_labels))
