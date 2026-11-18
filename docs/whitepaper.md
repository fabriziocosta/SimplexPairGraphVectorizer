# Simplex-Pair Graph Features

## Abstract

The vectorizer maps a labeled graph to counts of pairs of labeled clique patterns. Each occurrence starts from an unordered pair of representative vertices at a selected shortest-path distance. Every clique is assigned to exactly one representative, then cliques assigned to the two representatives are paired. The paired cliques need not share any vertex. At distance zero, both cliques come from the same representative's collection.

## Labeled graph and simplices

Let $G=(V,E,\ell_V,\ell_E)$ be a finite, simple, undirected graph with vertex and edge labels. A simplex is a nonempty clique, with dimension $|\sigma|-1$. Given a maximum dimension $q$, the allowed simplices are

$$\mathcal{C}_{q}(G)=\{\sigma\subseteq V:1\leq |\sigma|\leq q+1,\ \forall u,v\in\sigma,\ u\ne v\Rightarrow\{u,v\}\in E\}.$$

Each simplex $\sigma$ has one canonical representative $\rho(\sigma)$: its lowest vertex under a fixed stable node ordering. Its associated simplex collection is

$$\mathcal{C}_{q}(G;u)=\{\sigma\in\mathcal{C}_{q}(G):\rho(\sigma)=u\}.$$

Thus a clique belongs to exactly one representative's collection, even though its other vertices may be representatives of different cliques.

## Representative-marked simplex types

Write $\kappa_G(\sigma,u)$ for the type of simplex $\sigma$ represented by $u=\rho(\sigma)$. This type marks the representative's vertex label, hashes the labels of all simplex vertices, and hashes every induced edge together with the hashes of its two endpoint nodes. The node and edge signature lists are sorted before hashing, so the type does not depend on incidental iteration order.

## Representative pairs and feature occurrences

An unordered pair of representative vertices $\{u,v\}$ has unweighted shortest-path distance $d_G(u,v)$ in the full graph. It is eligible when

$$L\leq d_G(u,v)+\varepsilon\leq U,$$

where $\varepsilon$ is an optional nonnegative offset (zero by default). For each eligible representative pair, take the Cartesian product of $\mathcal{C}_q(G;u)$ and $\mathcal{C}_q(G;v)$. At distance zero, $u=v$ and the pair of simplices is unordered, with replacement when self-pairs are enabled. At positive distance, the two simplices may be disjoint or overlap. Since each clique is assigned once, a clique cannot be paired with itself across two different representatives. When self-pairs are disabled, identical cliques are omitted at distance zero.

The symbolic feature for an occurrence is

$$\phi_G(\{u,v\},\sigma,\tau)=\left(\left[\kappa_G(\sigma,u),\kappa_G(\tau,v)\right]_{\mathrm{unordered}},\ d_G(u,v)+\varepsilon\right).$$

The unordered pair of representative-marked simplex types is canonicalized by sorting its two members. This makes representative ordering immaterial: swapping $u,v$ and their associated simplices gives the same feature.

For a fixed representative $u$, this construction includes every eligible representative $v$ within the configured distance range and every clique assigned to each representative. In particular, when $u=v$, it pairs the cliques assigned to that single representative.

## Graph-level feature counts

Let $s\in\{0,1\}$ specify whether self-pairs are included, and let $\preceq$ be the stable ordering of vertices. Define $\mathrm{allowed}_s(\sigma,\tau)$ to be true for distinct simplices, and for identical simplices exactly when $s=1$. The count of feature $\phi$ is

$$c_G(\phi)=\sum_{u\preceq v\in V}\ \sum_{\sigma\in\mathcal{C}_q(G;u)}\ \sum_{\tau\in\mathcal{C}_q(G;v)}\mathbf{1}[L\leq d_G(u,v)+\varepsilon\leq U]\,\mathbf{1}[\phi=\phi_G(\{u,v\},\sigma,\tau)]\,\mathbf{1}[\mathrm{allowed}_s(\sigma,\tau)].$$

For $u=v$, the simplex-pair sum is unordered: each pair is counted once, with identical simplices allowed exactly when $s=1$. For $u\ne v$, the representative pair is visited once in a stable vertex order and the full Cartesian product is taken. Repeated feature types arising at different representative pairs add to the same count.

The resulting uncompressed representation is the count function $c_G:\mathcal{F}\rightarrow\mathbb{N}_0$. It is a sparse vector for any finite graph.

## Fixed-width vector representation

For learning, symbolic features are mapped to a fixed coordinate set of size $m=2^b$ by a deterministic hash $h$. The coordinate value is

$$x_G(j)=\sum_{\phi\in\mathcal{F},\ h(\phi)=j}c_G(\phi),\quad j=1,\ldots,m.$$

Hash collisions add their occurrence counts. The representation does not normalize counts or weight them by class.

## Protein contact graph interpretation

In the enzyme workflow, a graph vertex represents a C$\alpha$ atom from a selected protein chain, and an edge represents a residue pair within the selected spatial contact window. Vertex labels encode amino-acid categories, while edge labels encode contact labels or optional distance bins. The vectorizer forms representative pairs using shortest-path hops through this contact graph; these hops are distinct from spatial distances in ångströms.

## Interpretation and scope

Each coordinate aggregates evidence for pairs of locally labeled clique patterns assigned to representative vertices separated by a selected graph distance. The construction preserves node labels, induced edge labels, representative positions within cliques, and graph-scale separation. It is a summary rather than a lossless graph encoding: distinct graphs can yield the same counts, and the fixed-width hash can merge distinct symbolic features. Stable node ordering selects one representative for each clique and makes enumeration deterministic.
