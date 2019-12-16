import numpy as np
import time
from sklearn.metrics import pairwise_distances
import random

from HW5_FeatureVector import GenerateFeatures


class MinHashing:
    def __init__(self, data):
        self.data = data
        k_minhash = 16

        self.mh_jaccard = np.zeros((data.shape[0], data.shape[0]))

        rand_a = self.generate_random(k_minhash, self.data.shape[1])
        rand_b = self.generate_random(k_minhash, self.data.shape[1])

        nextPrime = self.nextPrime(self.data.shape[1])
        signature = np.full((k_minhash, self.data.shape[0]), np.inf)

        self.start_baseline = time.time()
        print("Baseline comparisons ...")
        self.baseline_cmp = 1 - pairwise_distances(self.data.toarray(), self.data.toarray(), 'jaccard')
        self.end_baseline = time.time()

        print("Signature generation ...")
        self.start_minhash = time.time()
        for col in range(self.data.shape[1]):
            colData = self.data.getcol(col).nonzero()[0]
            for values in colData:
                for k in range(k_minhash):
                    hashVal = ((rand_a[k] * col + rand_b[k]) % nextPrime) % self.data.shape[1]
                    signature[k, values] = min(signature[k, values], hashVal)
        self.end_sign = time.time()

        print("MinHash comparisons ...")
        for col1 in range(signature.shape[1]):
            for col2 in range(col1, signature.shape[1]):
                similar_sign = np.sum(np.equal(signature[:, col1], signature[:, col2]))
                self.mh_jaccard[col1, col2] = similar_sign / signature.shape[0]
                self.mh_jaccard[col2, col1] = self.mh_jaccard[col1, col2]
        self.end_minhash = time.time()

    def generate_random(self, k, maxShingleID):
        randNums = []
        while k > 0:
            randValues = random.randint(0, maxShingleID)
            while randValues in randNums:
                randValues = random.randint(0, maxShingleID)
            randNums.append(randValues)
            k -= 1
        return randNums

    def nextPrime(self, n):
        n += 1
        return n if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) else self.nextPrime(n)


if __name__ == "__main__":
    data = GenerateFeatures("sentiment labelled sentences")
    mh = MinHashing(data.data_matrix)

    print("Efficiency for baseline: ", mh.end_baseline - mh.start_baseline)
    print("Efficiency for MinHashing: ", mh.end_minhash - mh.start_minhash)
    print("Time taken to generate signatures: ", mh.end_sign - mh.start_minhash)

    # Compute Mean Squared Error
    mse = np.mean(np.square(mh.baseline_cmp - mh.mh_jaccard))
    print("Mean Squared Error: ", mse)