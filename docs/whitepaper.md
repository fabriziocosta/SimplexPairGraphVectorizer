# Simplex-Pair Graph Features

## Abstract

The vectorizer maps a labeled graph to counts of pairs of labeled clique patterns. Each occurrence starts from an unordered pair of anchor vertices at a selected shortest-path distance. Every clique containing one anchor can pair with every clique containing the other anchor; the cliques need not share any vertex. At distance zero, both clique collections are associated with the same anchor.

## Labeled graph and simplices

Let $G=(V,E,\ell_V,\ell_E)$ be a finite, simple, undirected graph with vertex and edge labels. A simplex is a nonempty clique, with dimension $|\sigma|-1$. Given a maximum dimension $q$, the allowed simplices are

$$\mathcal{C}_{q}(G)=\{\sigma\subseteq V:1\leq |\sigma|\leq q+1,\ \forall u,v\in\sigma,\ u\ne v\Rightarrow\{u,v\}\in E\}.$$

For each vertex $u$, its associated simplex collection is

$$\mathcal{C}_{q}(G;u)=\{\sigma\in\mathcal{C}_{q}(G):u\in\sigma\}.$$

Every graph vertex with at least one associated simplex can serve as an anchor. A clique is therefore represented once for each of its vertices, with that vertex marked as the anchor in the corresponding type.

## Anchored simplex types

Write $\kappa_G(\sigma,u)$ for the type of simplex $\sigma$ anchored at $u\in\sigma$. This type marks the anchor's vertex label, hashes the labels of all simplex vertices, and hashes every induced edge together with the hashes of its two endpoint nodes. The node and edge signature lists are sorted before hashing, so the type does not depend on incidental iteration order.

## Anchor pairs and feature occurrences

An anchor pair $\{u,v\}$ is unordered. Its distance is the unweighted shortest-path distance in the full graph, $d_G(u,v)$. It is eligible when

$$L\leq d_G(u,v)+\varepsilon\leq U,$$

where $\varepsilon$ is an optional nonnegative offset (zero by default). For each eligible anchor pair, take the Cartesian product of the associated simplex collections. At distance zero, $u=v$ and the pair of simplices is unordered, with replacement when self-pairs are enabled. At positive distance, a simplex containing $u$ is paired with a simplex containing $v$; the two simplices may be disjoint, overlap, or be identical if both contain both anchors. When self-pairs are disabled, identical simplex pairs are omitted.

The symbolic feature for an occurrence is

$$\phi_G(\{u,v\},\sigma,\tau)=\left(\left[\kappa_G(\sigma,u),\kappa_G(\tau,v)\right]_{\mathrm{unordered}},\ d_G(u,v)+\varepsilon\right).$$

The unordered pair of anchored simplex types is canonicalized by sorting its two members. This also makes anchor ordering immaterial: swapping $u,v$ and their associated simplices gives the same feature.

For a fixed anchor $u$, this construction includes every eligible anchor vertex $v$ within the configured distance range and every clique associated with each anchor. In particular, when $u=v$, it pairs the simplices associated with that single vertex.

## Graph-level feature counts

Let $s\in\{0,1\}$ specify whether self-pairs are included, and let $\preceq$ be the stable ordering of vertices. Define $\mathrm{allowed}_s(\sigma,\tau)$ to be true for distinct simplices, and for identical simplices exactly when $s=1$. The count of feature $\phi$ is

$$c_G(\phi)=\sum_{u\preceq v\in V}\ \sum_{\sigma\in\mathcal{C}_q(G;u)}\ \sum_{\tau\in\mathcal{C}_q(G;v)}\mathbf{1}[L\leq d_G(u,v)+\varepsilon\leq U]\,\mathbf{1}[\phi=\phi_G(\{u,v\},\sigma,\tau)]\,\mathbf{1}[\mathrm{allowed}_s(\sigma,\tau)].$$

For $u=v$, the simplex-pair sum is unordered: each pair is counted once, with identical simplices allowed exactly when $s=1$. For $u\ne v$, the anchor pair is visited once in a stable vertex order and the full Cartesian product is taken. Repeated feature types arising at different anchors or in different graph regions add to the same count.

The resulting uncompressed representation is the count function $c_G:\mathcal{F}\rightarrow\mathbb{N}_0$. It is a sparse vector for any finite graph.

## Fixed-width vector representation

For learning, symbolic features are mapped to a fixed coordinate set of size $m=2^b$ by a deterministic hash $h$. The coordinate value is

$$x_G(j)=\sum_{\phi\in\mathcal{F},\ h(\phi)=j}c_G(\phi),\quad j=1,\ldots,m.$$

Hash collisions add their occurrence counts. The representation does not normalize counts or weight them by class.

## Protein contact graph interpretation

In the enzyme workflow, a graph vertex represents a C$\alpha$ atom from a selected protein chain, and an edge represents a residue pair within the selected spatial contact window. Vertex labels encode amino-acid categories, while edge labels encode contact labels or optional distance bins. The vectorizer forms anchor pairs using shortest-path hops through this contact graph; these hops are distinct from spatial distances in ångströms.

## Interpretation and scope

Each coordinate aggregates evidence for pairs of locally labeled clique patterns anchored at vertices separated by a selected graph distance. The construction preserves node labels, induced edge labels, anchor positions within cliques, and graph-scale separation. It is a summary rather than a lossless graph encoding: distinct graphs can yield the same counts, and the fixed-width hash can merge distinct symbolic features. Stable node ordering makes enumeration deterministic; it does not restrict which vertices may anchor a clique.
