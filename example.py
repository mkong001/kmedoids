# coding: utf-8
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np

import kmedoids
import importpattern
import distmeasure

# Patterns in dataset
data = np.array(np.column_stack((importpattern.data, importpattern.data)))

# Alpha value to weigh ratios between two distance measures
alpha = .25

# Define custom dist function
def dfun(u, v):
    utemp = u[0].split(" ")
    vtemp = v[0].split(" ")

    usplit = []
    vsplit = []

    for eventu in utemp:
        usplit.append(eventu)
    for eventv in vtemp:
        vsplit.append(eventv)
    
    # u union v
    union = list(set(usplit) | set(vsplit))

    # Dist measure 1 based on Longest Common Subsequence
    dlcs = 1-(abs(distmeasure.lcs_length(usplit, vsplit))/abs(len(union)))

    # Dist measure 2 based on the occurrence frequency of each event
    docc = distmeasure.occ(usplit, vsplit, union)

    return (alpha * dlcs) + ((1-alpha) * docc)


# distance matrix
D = squareform(pdist(data, lambda u, v: dfun(u, v)))

# Silhouette algorithm
# param: 
#   * m --> medoids
#   # c --> clustering result
# returns: avg silhouette value
def silhouette(m, c):
    ssum = 0
    pointcount = 0
    for label in c:
        # |C(i)| - # of data points in cluster assigned to this label
        clength = abs(len(c[label]))

        # add |C(i)| to total count of points in dataset
        pointcount = pointcount + clength

        # check that |C(i)| is not 1.
        if clength != 1:
            # For each data point at point_idx in cluster c[label],
            for point_idx in c[label]:
                # a(i) - measure how well assigned the ith data point is to its cluster
                # by calculating the mean distance between i and all other points in
                # the same cluster
                dsum_a = 0
                # distance between data points i and j in cluster c[label]
                for j_a in c[label]:
                    if point_idx != j_a:
                        dsum_a = dsum_a + dfun(data[point_idx], data[j_a])

                a = (1/(float(clength-1)))*(dsum_a)

                # b(i) - measure the smallest mean distance of i to all points in
                # any other cluster, of which i is not a member
                b = 0
                for k in c:
                    if label != k:
                        dsum_b = 0
                        for j_b in c[k]:
                            dsum_b = dsum_b + dfun(data[point_idx], data[j_b])
                        davg_b = (1/(float(len(c[k]))))*(dsum_b)
                        if b > davg_b:
                            b = davg_b
            
                # s(i) - silhouette (value) for data point i
                si = (b - a)/(float(max(a,b)))
                # add s(i) to total s (ssum)
                ssum = ssum + si       

    return ssum/(float(pointcount))

    
range_n_clusters = [10, 20, 30, 40, 50]

for n_clusters in range_n_clusters:
    # split into n_clusters clusters
    M, C = kmedoids.kMedoids(D, n_clusters)

    # calculate avg silhouette value
    sscore = silhouette(M, C)
    #sscore = silhouette_score(D, C, metric="precomputed")
    print("score for cluster count {}: {}".format(n_clusters, sscore))

    # print('medoids:')
    # for point_idx in M:
    #     print( data[point_idx][0] )

    # print('')
    # print('clustering result:')
    # for label in C:
    #     #print(abs(len(C[label])))
    #     for point_idx in C[label]:
    #         print('label {0}:　{1}'.format(label, data[point_idx][0]))
            # print(dfun(data[point_idx], data[point_idx+1]))
            #print(dfun(data[point_idx], data[point_idx]))



# split into 20 clusters
M, C = kmedoids.kMedoids(D, 20)

print('medoids:')
for point_idx in M:
    print( data[point_idx][0] )

print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, data[point_idx][0]))


