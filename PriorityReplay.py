import math

alpha = 1.01
N = 10  # len(memory)


# Partial sum of power series https://www.wolframalpha.com/input?i=1+%2B+x+%2B+x%5E2+%2B+...+series
def index_cumulative(index):
    return index + 1 if alpha == 1 \
        else (alpha ** (1 + index) - 1) / (alpha - 1)


total = index_cumulative(N - 1)


# Derived from rand = index_cumulative(index) / index_cumulative(total)
def prioritized_index(rand):  # rand is a random value between 0 and 1.
    return max(0, round(math.log(rand * total * (alpha - 1) + 1, alpha) - 1))  # Return index


# Derived from: size of segment = index_cumulative(index) / index_cumulative(total)
# (N * proba(index)) ** -beta / [(N * proba(N - 1)) ** -beta] = importance sampling weight to multiply in loss term
def proba(index):  # Proba of sampling that index
    return (index_cumulative(index) - index_cumulative(index - 1)) / total


for r in [0, 0.05, 0.1, 0.2, 0.3, 1]:
    i = prioritized_index(r)
    print(f'index for random float {r}: {i}, proba of index: {proba(i)}')  # O(1) read/sample complexity


# Pareto distribution? https://en.wikipedia.org/wiki/Pareto_distribution


# Use pyskiplist for sorted inserts/replace/deletes. - Never mind
# O(log(N)) sampling, O(log(N)) insert/replace/delete. - or use sorted list for offline. - Never mind
# Maybe episodes can have a running tally for prioritization; this aggregate gets sampled; then it has another sampler
# related https://arxiv.org/pdf/1905.12726.pdf decays the priority to earlier experiences in the episode
# Note: importance sampling proba now depends on product of two probas and denominator term becomes estimate N, then M
# Time complexity doesn't change O(log(NM) = log(N) + log(M)) for episodes N and avg num experiences per episode M. - NM
# Or O(N + M) rather than O(NM) for bisect.insort. If N approx M, then this gives O(1) sampling, and O(sqrt(NM)) insert.
# MuJoCo M = 1000, thus optimal at capacity of 1e6, which is what DrQV2 used for SotA.
# O(1) for offline - no downside except maybe writing


# Here is how PER did it:
#  In this case, P becomes a power-law distribution with exponent α.
#  To efficiently sample from distribution (1), the complexity cannot depend on N.
#  For the rank-based variant, we can approximate the cumulative density function
#  with a piecewise linear function with k segments of equal probability.
#  The segment boundaries can be precomputed (they change only when N or α change).
#  At runtime, we sample a segment, and then sample uniformly among the transitions within it.
#  This works particularly well in conjunction with a minibatch-based learning algorithm: choose
#  k to be the size of the minibatch, and sample exactly one transition from each segment – this is
#  a form of stratified sampling that has the added advantage of balancing out the minibatch
#  (there will always be exactly one transition with high magnitude δ, one with medium magnitude, etc).
# Requires slightly more memory for storing the segments. Sampling w/o replacement is about O(N).
# Creating the segments every time data is added (N changes) is also expensive.


# Better answer https://stats.stackexchange.com/a/406664
#  Construct segments of similar priority span and sample zipf
#  https://github.com/philiptkd/DQN/blob/master/zipf.py


# I could change alpha to fit a running tally of the priority distribution
# bisect.insort segments is actually faster. Can add to segments and occasionally rebalance by slicing and merging.
# Can use above to sample from segments and episodes.


# More advanced
# Divide into k percentiles
# Keep track of percentiles with each add/rewrite based on running statistics of total, count, and added
# Power-law distribution w.r.t. the k percentiles
# Percentiles like centroids for clustering
#       For example, the median increments to the next index when two values of larger magnitude are added
#       Can keep track for each percentile
#           keep count (-2, 2, 0, 1, or 2) increment/decrement index/update value when -2 or 2; reset to 0
#           O(k) for adding, then O(N/k) for adding, then O(M) for adding
#           Not counting list insert, it's O(log...) for comparing. Percentile updates may be vectorized, still O(k).
#       Then sample from k via power-law (e.g. the above)
#       And then sample from that segment via power-law (e.g. the above)
#       Then sample experience via power-law (e.g. the above)

# No need for large data structures and re-balancing (see time complexities of sum-trees (log(MN) for both add/sample)
# and segment trees (...) respectively)

# This might generalize to d dimensions if centroids are d-dimensional. Have to compute stats for each d_i and maintain
# each sorting combination.
