import numpy as np

def find_max_clique(pnts1, pnts2, dist_thrs=0.2):
    # in-lier detection algorithm
    n_pnts = pnts1.shape[0]
    W = np.zeros((n_pnts, n_pnts))

    count = 0
    maxn = 0
    maxc = 0
    # diff of pairwise euclidean distance between same points in T1 and T2
    for i in range(numPoints):
        diff_1 = pnts1[i,:] - pnts1
        diff_2 = pnts2[i,:] - pnts2
        dist_1 = np.linalg.norm(diff_1, axis=1)
        dist_2 = np.linalg.norm(diff_2, axis=1)
        diff = abs(dist_2 - dist_1)
        wIdx = np.where(diff < dist_thrs)
        W[i,wIdx] = 1
        count = np.sum(W[i,:])
        if count > maxc:
            maxc = count
            maxn = i
        count=0

    clique = [maxn]
    isin = True

    while True:
        potentials = list()
        # Find potential nodes which are connected to all nodes in the clique
        for i in range(numPoints):
            Wsub = W[i, clique]
            sumForIn = np.sum(Wsub)
            if sumForIn == len(clique):
                isin = True
            else:
                isin = False

            if isin == True and i not in clique:
                potentials.append(i)
            isin=True

        count = 0
        maxn = 0
        maxc = 0
        # Find the node which is connected to the maximum number of potential nodes and store in maxn
        for i in range(len(potentials)):
            Wsub = W[potentials[i], potentials]
            count = np.sum(Wsub)

            if count > maxc:
                maxc = count
                maxn = potentials[i]
            count = 0
        if maxc == 0:
            break
        clique.append(maxn)

        if (len(clique) > 100):
            break

    return clique