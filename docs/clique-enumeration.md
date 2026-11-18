# Clique enumeration strategies

The vectorizer enumerates every clique in the requested simplex-dimension
range once, then assigns each clique to its canonical representative. The
selected enumerator changes how this clique set is generated; it does not
change the feature definition or the resulting vector.

## Choosing an enumerator

Set `clique_enumerator` on `SimplexPairGraphVectorizer`:

| Value | Behavior |
| --- | --- |
| `"auto"` | Default. Use the bitset implementation when the maximum requested simplex dimension is 0–3; use NetworkX for larger dimensions. |
| `"fast"` | Force the bitset adjacency-intersection implementation. It supports maximum simplex dimension 3. |
| `"generic"` | Force NetworkX's generic all-clique enumeration at any supported dimension. |

For example, `simplex_dimension=(1, 3)` is within the fast range, while
`simplex_dimension=(1, 4)` selects the generic enumerator with `"auto"`.
Dimension bounds are inclusive. Requesting `"fast"` above dimension 3 raises a
`ValueError`; use `"auto"` to fall back automatically or `"generic"` to force
NetworkX.

Both implementations use the same stable node ordering and enumerate each
clique only once. The bitset implementation records each vertex's neighbors
with higher stable ordering as an integer bit mask. It extends a clique by
intersecting the remaining candidates with each added vertex's higher-neighbor
mask. This generates each bounded clique once without scanning NetworkX's
complete clique stream. The generic implementation uses
`networkx.enumerate_all_cliques` and stops after the requested maximum size.

## Protein-graph timing experiment

Notebook 06 compares these same implementations on five cached protein
contact graphs, with dimensions 0–3. Each graph's complete clique sets were
checked for equality before timing. With five repetitions per method, the
median times summed across the five graphs were 11.12 ms for NetworkX and
3.51 ms for the bitset implementation, a 3.17× aggregate speedup. The sampled
graphs ranged from 156 to 512 vertices and contained 1,502 to 3,667 cliques in
the measured range.

These timings cover clique enumeration only. Representative-vertex distance
search and the later pairing and hashing work are not included, so total
vectorization speedup can differ.
