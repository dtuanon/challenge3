from sklearn.cluster import KMeans

def cluster_videos_kmeans(X, names, n_clusters = 970):
	kmeans			= KMeans(n_clusters=n_clusters, n_jobs = -1)
	
	belonging_index	= kmeans.fit_predict(X)
	clusters		= [set() for _ in range(n_clusters)]
	for cluster_idx, name in zip(belonging_index, names):
		clusters[cluster_idx].add(name)
	
	return clusters
