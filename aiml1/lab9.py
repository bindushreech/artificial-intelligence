import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):

        # Randomly select initial centroids
        indices = np.random.choice(
            len(X),
            self.k,
            replace=False
        )

        centroids = X[indices]

        for _ in range(self.max_iters):

            # Assign each point to the nearest centroid
            labels = []

            for point in X:
                distances = [
                    euclidean_distance(point, centroid)
                    for centroid in centroids
                ]

                labels.append(np.argmin(distances))

            labels = np.array(labels)

            # Calculate new centroids
            new_centroids = []

            for i in range(self.k):

                cluster_points = X[labels == i]

                if len(cluster_points) > 0:
                    new_centroids.append(
                        cluster_points.mean(axis=0)
                    )
                else:
                    new_centroids.append(centroids[i])

            new_centroids = np.array(new_centroids)

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

        return labels


def silhouette_score_simple(X, labels):

    scores = []

    for i in range(len(X)):

        same_cluster = X[labels == labels[i]]

        # Distance within the same cluster
        if len(same_cluster) > 1:
            a = np.mean([
                euclidean_distance(X[i], point)
                for point in same_cluster
                if not np.array_equal(X[i], point)
            ])
        else:
            a = 0

        # Distance to the nearest other cluster
        b_values = []

        for cluster in set(labels):

            if cluster != labels[i]:

                other_cluster = X[labels == cluster]

                distance = np.mean([
                    euclidean_distance(X[i], point)
                    for point in other_cluster
                ])

                b_values.append(distance)

        b = min(b_values)

        if max(a, b) == 0:
            score = 0
        else:
            score = (b - a) / max(a, b)

        scores.append(score)

    return np.mean(scores)


if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [1, 4],
        [1, 0],
        [10, 2],
        [10, 4],
        [10, 0]
    ])

    kmeans = KMeans(k=2)

    labels = kmeans.fit(X)

    print("Cluster Labels:")
    print(labels)

    score = silhouette_score_simple(X, labels)

    print("Silhouette Score:")
    print(score)