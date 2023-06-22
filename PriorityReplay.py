import math

alpha = 1.01
N = 10  # len(memory)


def index_cumulative(index):
    return index + 1 if alpha == 1 \
        else (alpha ** (1 + index) - 1) / (alpha - 1)


total = index_cumulative(N - 1)


def prioritized_index(rand):  # rand is a random value between 0 and 1.
    return max(0, round(math.log(rand * total * (alpha - 1) + 1, alpha) - 1))


# (N * proba(index)) ** -beta / [(N * proba(N - 1)) ** -beta] = importance sampling weight to multiply in loss term
def proba(index):  # Proba of sampling that index
    return (index_cumulative(index) - index_cumulative(index - 1)) / total


for r in [0, 0.05, 0.1, 0.2, 0.3, 1]:
    i = prioritized_index(r)
    print(f'index for random float {r}: {i}, proba of index: {proba(i)}')  # O(1) read/sample complexity

# Use pyskiplist for sorted inserts/replace/deletes. - Never mind
# O(log(N)) sampling, O(log(N)) insert/replace/delete. - or use sorted list for offline. - Never mind
# Maybe episodes can have a running tally for prioritization; this aggregate gets sampled; then it has another sampler
# related https://arxiv.org/pdf/1905.12726.pdf decays the priority to earlier experiences in the episode
# Note: importance sampling proba now depends on product of two probas and denominator term becomes estimate N, then M
# Time complexity doesn't change O(log(NM) = log(N) + log(M)) for episodes N and avg num experiences per episode M. - NM
# Or O(N + M) rather than O(NM) for bisect.insort. If N approx M, then this gives O(1) sampling, and O(sqrt(NM)) insert.
# MuJoCo M = 1000, thus optimal at capacity of 1e6, which is what DrQV2 used for SotA.
# O(1) for offline - no downside except maybe writing