import numpy as np
import os


def humanize_number(value, fraction_point=1):
    powers = [10 ** x for x in (12, 9, 6, 3, 0)]
    human_powers = ('T', 'B', 'M', 'K', '')
    is_negative = False
    if not isinstance(value, float):
        value = float(value)
    if value < 0:
        is_negative = True
        value = abs(value)
    for i, p in enumerate(powers):
        if value >= p:
            return_value = str(int(round(value / (p / (10.0 ** fraction_point))) /
                                   (10 ** fraction_point))) + human_powers[i]
            break
    if is_negative:
        return_value = "-" + return_value

    return return_value


def generate_points_ndim(width, size, mindist=10.0):
    points = [np.array([np.random.randint(0, w) for w in width])]
    while (len(points) < size):
        newpoint = np.array([np.random.randint(0, w) for w in width])
        # Calculate distances to all other points
        dist = [np.linalg.norm(newpoint - p) for p in points]
        # Only add this point if it is far enough away from all others.
        if (all([newd > mindist for newd in dist])):
            points.append(newpoint)
    return points


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


def generate_fixed_sum_sequence(n, sum_):
    return np.random.multinomial(sum_, np.ones(n) / n, size=1)[0]


def generate_Ndimensional_clusters(prefix, Xwidth,
                                   var_num=3,
                                   cluster_number=10,
                                   # number_of_points = 1000000000,
                                   minpoints_per_cluster=5000,
                                   maxpoints_per_cluster=9000,
                                   mindist=4.5,
                                   maxcov=2.0):
    # Create data clusters from  different multivariate distributions
    # point_numbers_per_cluster = generate_fixed_sum_sequence(cluster_number, number_of_points)

    if len(Xwidth) != var_num:
        print("Variable borders do not match number of variables!")
        return
    means = generate_points_ndim(Xwidth, cluster_number, mindist=mindist)
    print("Center points generated")
    all_size = 0
    prefix += "_" + str(cluster_number) + "cl_X" + str(var_num) + \
              "_" + humanize_number(minpoints_per_cluster) + "_" + humanize_number(maxpoints_per_cluster)
    tempname = prefix + "_XID_TMP.csv"
    XIDfile = open(tempname, 'w+')
    for i in range(cluster_number):
        cov = np.zeros((var_num, var_num))
        for j in range(var_num):
            for k in range(var_num):
                if j == k:
                    cov[j, k] = np.random.uniform(1, maxcov)
                else:
                    cov[j, k] = 0
        size = np.random.randint(minpoints_per_cluster, maxpoints_per_cluster)  # point_numbers_per_cluster[i]
        all_size += size

        cluster = np.random.multivariate_normal(mean=means[i], cov=cov, size=size)
        clX = []
        for j in range(var_num):
            clX.append(cluster[:, j])
        clX.append([i] * len(clX[0]))

        np.savetxt(XIDfile, np.array(clX).T, delimiter=",")  # np.vstack((clusterX,clusterY,[i]*len(clusterX))).T

    XIDfile.close()
    print("Number of points: " + str(all_size))
    os.rename(tempname, prefix + "_" + str(all_size) + "_XID.csv")


def generate_clusters(prefix,
                      cluster_number=50,
                      # number_of_points = 1000000000,
                      minpoints_per_cluster=50000,
                      maxpoints_per_cluster=90000,
                      mindist=3.5,
                      maxcov=2.0,
                      size_X=70,
                      size_Y=70):
    # Create data clusters from  different multivariate distributions
    # point_numbers_per_cluster = generate_fixed_sum_sequence(cluster_number, number_of_points)
    means = generate_points(size_X, size_Y, cluster_number, mindist=mindist)
    print("Number of clusters: " + str(cluster_number))
    print("Center points generated")
    all_size = 0
    prefix += "_" + str(cluster_number) + "cl_" + \
              humanize_number(minpoints_per_cluster) + "_" + humanize_number(maxpoints_per_cluster)
    tempname = prefix + "_XID_TMP.csv"
    XIDf = open(tempname, 'w+')
    for i in range(cluster_number):
        cov = [[np.random.uniform(1, maxcov), 0], [0, np.random.uniform(1, maxcov)]]
        size = np.random.randint(minpoints_per_cluster, maxpoints_per_cluster)  # point_numbers_per_cluster[i]
        all_size += size
        cluster = np.random.multivariate_normal(mean=means[i], cov=cov, size=size)
        clusterX = cluster[:, 0]
        clusterY = cluster[:, 1]
        np.savetxt(XIDf, np.vstack((clusterX, clusterY, [i] * len(clusterX))).T, delimiter=",")
        print("Cluster " + str(i) + " saved")

    XIDf.close()
    print("Number of points: " + str(all_size))
    os.rename(tempname, prefix + "_" + str(all_size) + "_XID.csv")


# generate_clusters("./data_simulation/Cluster")

generate_Ndimensional_clusters("./data_simulation/Cluster", [50, 50, 50, 50], var_num=4)
