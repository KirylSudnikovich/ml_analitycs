import numpy as np


def generate_points(X_width, Y_width, size, mindist=10.0):
    points = [[np.random.randint(0, X_width), np.random.randint(0, Y_width)]]

    d2 = mindist ** 2
    while (len(points) < size):
        newpoint = [np.random.randint(0, X_width), np.random.randint(0, Y_width)]
        # Calculate squared distances to all other points
        dist2 = [abs(newpoint[0] - p[0]) ** 2 + abs(newpoint[1] - p[1]) ** 2 for p in points]
        # Only add this point if it is far enough away from all others.
        if (all([newd2 > d2 for newd2 in dist2])):
            points.append(newpoint)
    return points


def generate_clusters(prefix,
                      cluster_number=6,
                      minpoints_per_cluster=1000,
                      maxpoints_per_cluster=8000,
                      mindist=4.5,
                      maxcov=2.0,
                      size_X=15,
                      size_Y=15):
    # Create data clusters from  different multivariate distributions
    clusters = []
    means = generate_points(size_X, size_Y, cluster_number, mindist=mindist)
    all_size = 0
    for i in range(cluster_number):
        cov = [[np.random.uniform(1, maxcov), 0], [0, np.random.uniform(1, maxcov)]]
        size = np.random.randint(minpoints_per_cluster, maxpoints_per_cluster)
        all_size += size
        clusters.append(np.random.multivariate_normal(mean=means[i], cov=cov, size=size))
    print("Number of points: " + str(all_size))

    # Convert data to lists of X and Y, with and without cluster division
    # TODO: Cleanup, remove X,Y and maybe clX,clY and make Clusters_ShowPlot compatible
    X = []
    Y = []
    clX = []
    clY = []
    XYID = []
    i = 0
    for cluster in clusters:


        clusterX = cluster[:, 0]
        clusterY = cluster[:, 1]
        X.extend(clusterX)
        Y.extend(clusterY)
        XYID.extend(np.vstack((clusterX,clusterY,[i]*len(clusterX))).T)
        clX.append(clusterX)
        clY.append(clusterY)
        i+=1

    # save result

    # version with cluster separation for visualization
    np.savez(prefix + "_viz.npz", clX=np.asarray(clX), clY=np.asarray(clY))

    # X and Y separately
    np.savetxt(prefix + "_X.csv", X, delimiter=",", header="X")
    np.savetxt(prefix + "_Y.csv", Y, delimiter=",", header="Y")

    # X Y clusterID
    np.savetxt(prefix + "_XYID.csv", XYID, delimiter=",", header="X,Y,Label")


generate_clusters("./data_simulation/Cluster5")