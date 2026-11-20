# CNA Matrix Factorization via Minimum-Cost Flows and Perfect Phylogeny

A Python implementation of a graph-based heuristic for factoring an observed copy-number alteration (CNA) matrix into a perfect-phylogeny matrix and a consecutive-ones matrix:

\[
C = AB.
\]

For an input matrix \(C \in \mathbb{Z}_{\ge 0}^{n \times k}\), the goal is to construct:

- **\(A \in \{0,1\}^{n \times q}\)**, a rooted perfect-phylogeny matrix. The supports of any two columns of \(A\) are either disjoint or nested.
- **\(B \in \{0,1\}^{q \times k}\)**, a consecutive-ones matrix in which each row contains a single contiguous block of ones.

Each row of \(B\) represents an interval mutation, and \(A_{ij}=1\) indicates that mutation \(j\) is present in taxon \(i\). The number of inferred mutations, \(q\), is determined by the algorithm.

> **Status:** This is a heuristic implementation, not a guaranteed optimal factorization algorithm. The program explicitly checks whether the returned matrices satisfy all three requirements: \(C=AB\), perfect phylogeny of \(A\), and the consecutive-ones property of \(B\).

## Algorithm

### 1. Build a directed, weighted graph

Create one vertex for each observed row \(C_i\) of \(C\), including an all-zero row representing the root. **The zero row must be present in the input; `build_graph` does not insert it automatically.**

For each ordered pair of distinct vertices \((C_j,C_i)\), consider the difference

\[
x = C_i-C_j.
\]

An edge \(C_j \to C_i\) is eligible only when \(x\) is componentwise nonnegative. Its weight is the minimum number of distinct consecutive-ones vectors needed to represent \(x\), found using a minimum-cost flow formulation.

For \(x=(x_1,\ldots,x_k)\), define the discrete derivative

\[
\partial x=(x_1,\,x_2-x_1,\,\ldots,\,x_k-x_{k-1},\,-x_k).
\]

Construct a breakpoint graph with vertices \(1,\ldots,k+1\). Every arc \(p\to q\) with \(p<q\) represents the inclusive interval \([p,q-1]\), has capacity one, and has cost one. The optimization is

\[
\begin{aligned}
\min\quad & \sum_{1\le p<q\le k+1} f_{pq}\\
\text{subject to}\quad &
\sum_{q>i} f_{iq}-\sum_{p<i}f_{pi}=(\partial x)_i,
&& i=1,\ldots,k+1,\\
&0\le f_{pq}\le 1,
&&1\le p<q\le k+1.
\end{aligned}
\]

NetworkX solves this as a minimum-cost flow using node demands \(-\partial x\). With integral demands and capacities, the selected arcs yield a decomposition of \(x\) into consecutive-ones vectors. If the flow is infeasible, the candidate edge is omitted; otherwise, its minimum cost becomes the edge weight in \(G\).

### 2. Compute a minimum-cost rooted spanning tree

Find a minimum-cost spanning **arborescence** \(T\) of \(G\), with edges directed away from the root. All graph edges outside \(T\) are discarded. The implementation requires a unique indegree-zero vertex and checks that every vertex is reachable from it.

### 3. Label tree edges with interval mutations

Process the edges of \(T\) in nondecreasing order of their original weights. For each tree edge \(C_j\to C_i\), recompute a minimum-cost interval decomposition of \(C_i-C_j\), subject to previously assigned mutations.

Once an interval has been used on an edge **leaving \(C_j\)**, that interval may be reused on other edges leaving \(C_j\), but not on edges with a different source. The implementation tracks this restriction with a `mutation_owner` dictionary. If an edge cannot be decomposed under these restrictions, this stage fails.

### 4. Enforce local perfect-phylogeny structure

For each internal vertex \(C_j\), construct a local binary matrix \(A_j\):

- Rows represent \(C_j\) (an all-zero local root) and its immediate children.
- Columns represent the distinct interval mutations on edges leaving \(C_j\).
- An entry is one when the corresponding child edge carries that mutation.

A greedy canonicalization chooses mutations based on their support within the current row partition. It changes conflicting one-entries to zero, producing locally compatible mutation supports.

For each affected child, the removed mutation vectors are summed and **re-decomposed** using intervals not already used anywhere in the current phylogeny. The replacement intervals have the same sum as the removed ones, so successful repairs preserve the corresponding CNA difference. Repairs can also be infeasible.

### 5. Assemble and verify the global factorization

After local repairs:

1. Create one column of \(A\) for each distinct inferred interval mutation.
2. Create the corresponding row of \(B\) as its consecutive-ones indicator vector.
3. Set each row of \(A\) from the mutations accumulated along the root-to-taxon path in \(T\).
4. Check **exact reconstruction** (`C == A @ B`), the rooted perfect-phylogeny condition on \(A\), and the consecutive-ones condition on \(B\).

The test driver prints `SUCCESS` and the resulting matrices only if all checks pass; otherwise, it prints `FAIL` (with additional diagnostic checks if final verification fails).

## Requirements

- Python 3
- NumPy
- NetworkX

Install the dependencies:

```bash
pip install numpy networkx
```

## Running the script

The supplied file runs a synthetic test **immediately when executed**; it does not currently provide a command-line interface.

```bash
python "cna_script_global_factorization(1).py"
```

If you rename the file, adjust the command accordingly.

By default, the test uses:

```python
n, m, k = 20, 40, 80

A_true = generate_perfect_phylogeny(n, m)
B_true = generate_consecutive_ones(m, k)
C = A_true @ B_true
```

Here, `n` is the number of taxa (observed rows), `m` is the number of mutations in the **synthetic ground truth**, and `k` is the number of genomic positions. The algorithm does **not** require the recovered factorization to use exactly `m` mutations.

To test a different synthetic size, change `n`, `m`, and `k` in the test section at the bottom of the script. To use an observed matrix instead, replace the synthetic-data assignments with, for example:

```python
C = np.array([
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
], dtype=int)
```

Leave the subsequent calls to `build_graph`, `minimum_cost_spanning_tree`, `compute_mutation_labels`, `apply_greedy_repairs`, and `build_global_factorization` in place.

**Input assumptions:** `C` should be a two-dimensional nonnegative integer matrix that includes an all-zero root row. The current method uses componentwise nonnegative CNA differences along directed tree edges; it does not model copy-number losses or a nonzero baseline directly. Repeated identical rows and other degenerate inputs are not specially handled.

## Synthetic data generation

Two functions construct known-valid test instances:

```python
generate_perfect_phylogeny(n, m, seed=None)
generate_consecutive_ones(m, k, seed=None)
```

- `generate_perfect_phylogeny` incrementally constructs a binary perfect phylogeny with `m` mutation columns, retains an all-zero row, and samples down to `n` rows (`1 <= n <= m + 1`).
- `generate_consecutive_ones` samples `m` distinct nonempty intervals from `k` positions (`m <= k(k + 1)/2`), with one interval per row.

Their product supplies the observed input `C = A_true @ B_true`. Because factorizations need not be unique, validation compares the recovered product and structural properties rather than requiring `A` and `B` to match the ground-truth factors exactly.

## Main functions

| Function | Purpose |
| --- | --- |
| `discrete_derivative(x)` | Compute \(\partial x\). |
| `minimum_interval_cost(x)` | Return the minimum interval-decomposition cost, or `None`. |
| `build_graph(C)` | Build the weighted directed graph \(G\). |
| `minimum_cost_spanning_tree(G)` | Compute a minimum-cost rooted arborescence \(T\). |
| `compute_mutation_labels(T, C)` | Assign interval mutations to tree edges subject to source ownership. |
| `build_local_binary_matrices(labeled_T, C)` | Construct local mutation-incidence matrices. |
| `canonicalize_greedy(A)` | Greedily correct a local binary matrix toward perfect-phylogeny structure. |
| `apply_greedy_repairs(labeled_T, C)` | Re-decompose conflicting mutations and update tree labels. |
| `build_global_factorization(repaired_T, C)` | Return `(A, B, mutation_order)`, or `None` if the root is not all-zero. |
| `is_perfect_phylogeny_matrix(A)` | Check whether column supports are nested or disjoint. |
| `is_consecutive_ones_matrix(B)` | Check whether every row is binary and has consecutive ones. |

`mutation_order` contains the inferred intervals as **1-based, inclusive** `(start, end)` pairs. Rows of `A` preserve the row indexing of the input `C`, and rows of `B` follow `mutation_order`.

## Limitations

- **Heuristic search:** The minimum-cost arborescence optimizes the preliminary graph weights, but the later mutation assignment and local repairs are greedy. The algorithm does not guarantee a globally minimum-size factorization or success on every valid input.
- **Possible infeasibility:** Capacity-one interval decompositions, source-specific mutation ownership, and the requirement for previously unused repair intervals can cause the pipeline to return `FAIL`.
- **Scalability:** Graph construction considers every ordered pair of observed rows and solves flow problems over up to \(O(k^2)\) potential interval arcs per pair. Large matrices can therefore be costly.
- **Input conventions:** The present implementation assumes nonnegative integer copy numbers, an explicit all-zero root, and additive interval mutations. It has no command-line argument parser, packaged API, or automated test suite beyond the executable synthetic test.

## Interpretation

A successful result gives an **exact** decomposition of the observed matrix into a phylogenetically compatible binary mutation-incidence matrix and a binary interval matrix. The inferred factorization is one feasible explanation of the data, not necessarily the original generating factorization or the one using the fewest possible mutations.
