# Feature hashing and serialization

## Current feature hash path

The vectorizer hashes representative-marked node labels, induced edges and
their endpoint hashes, and then simplex pairs with their representative
distance. Unordered node, edge, and simplex hash collections are sorted before
they are combined. BLAKE2b remains the digest algorithm for these feature
hashes and for mapping feature payloads to vector coordinates.

Canonical payloads are normalized into type-tagged values and serialized with
`orjson` before hashing. This changes the JSON encoder, not the hash algorithm
or feature definition. The package falls back to Python's standard JSON
encoder when `orjson` cannot encode a value, such as an integer outside its
supported range.

## Comparison with AbstractGraph SHA-256

Notebook 08 compares three implementations on the same cached protein graphs:

| Hashing and serialization | Sum of per-graph median times |
| --- | ---: |
| Current BLAKE2b with Python's standard JSON encoder | 2.87 s |
| Current BLAKE2b with `orjson` | 1.13 s |
| AbstractGraph-style SHA-256 with canonical typed values | 1.54 s |

The run used three graphs, dimensions 0–3, representative distances 0–1, and
three timing repetitions per method. `orjson` was about 2.53× faster than the
standard encoder and about 1.36× faster than the AbstractGraph-style SHA-256
path. The `orjson` path produced the same feature coordinates as the standard
encoder. The SHA-256 variant produced the same total feature occurrence count,
but different hash coordinates as expected from its different digest and
encoding.

These are timings for a small sample of protein graphs and may vary by machine
and dataset. The benchmark confirms the relative performance of these hash
paths for this workload; it does not estimate total training time.
