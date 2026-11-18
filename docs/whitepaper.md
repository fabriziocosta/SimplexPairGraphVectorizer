# Simplex-Pair Graph Features

## Abstract

This note defines the graph representation used by the simplex-pair vectorizer. A labeled graph is mapped to a nonnegative vector of occurrence counts. Each count describes two labeled clique patterns that share a chosen root vertex, together with the graph distance between representatives of the two cliques. The construction combines local labeled structure with a bounded measure of separation in the full graph.

## Labeled graph and simplices

Let

$G=(V,E,\ell_V,\ell_E)$

be a finite, simple, undirected graph. Here $V$ is the vertex set, $E$ is the edge set, and $\ell_V$ and $\ell_E$ assign labels to vertices and edges. The label sets need not be numeric.

A simplex of $G$ is a clique: a nonempty set of vertices $\sigma\subseteq V$ for which every distinct pair in $\sigma$ is an edge. Its dimension is

$\dim(\sigma)=|\sigma|-1$

Thus vertices, edges, triangles, and four-vertex cliques have dimensions 0, 1, 2, and 3 respectively. Given a maximum simplex dimension $q$, the allowed simplex family is

$\mathcal{C}_{q}(G)=\{\sigma\subseteq V:1\leq |\sigma|\leq q+1,\ \forall u,v\in\sigma,\ u\ne v\Rightarrow\{u,v\}\in E\}$

For a root vertex $r$, the simplices containing that root are

$\mathcal{C}_{q}(G;r)=\{\sigma\in\mathcal{C}_{q}(G):r\in\sigma\}$

## Rooted simplex types

The root distinguishes one vertex within a simplex. Write $\kappa_G(\sigma,r)$ for the labeled type of the rooted simplex $(\sigma,r)$. This type records the vertex labels on $\sigma$ with the root marked, the edge labels on every edge induced by $\sigma$, and the incidence of those labels within the clique.

Consequently, a rooted simplex type represents a small, labeled graph pattern, not just a list of amino-acid labels. The root mark retains which vertex is shared when two simplex patterns are paired. A consistent ordering convention for vertices is used when assigning these types and choosing representatives.

## Rooted pairs and representative distance

A feature occurrence starts with two simplices $\sigma$ and $\tau$ that both contain a root $r$. The pair is unordered: swapping $\sigma$ and $\tau$ does not create a distinct feature. The pair may contain the same simplex twice when self-pairs are enabled. If $\sigma$ and $\tau$ share more than one vertex, their pairing is considered once for each shared root.

Each simplex has a representative vertex $\rho(\sigma)$, selected as its least vertex under the fixed vertex ordering. The pair distance is the shortest path length in the full graph between these representatives:

$\delta_G(\sigma,\tau)=d_G\bigl(\rho(\sigma),\rho(\tau)\bigr)$

Here $d_G$ is ordinary unweighted graph distance. An optional nonnegative offset $\varepsilon$ may be added before filtering; with discrete graph distances the default offset is zero. A pair contributes only when

$L\leq\delta_G(\sigma,\tau)+\varepsilon\leq U$

These distance bounds are measured in graph edges (contact-network hops), not in ångströms. For a self-pair, the representative distance is zero.

The symbolic feature associated with an eligible rooted pair is

$\phi_G(r,\sigma,\tau)=\bigl((\min_{\preceq}\{\kappa_G(\sigma,r),\kappa_G(\tau,r)\},\ \max_{\preceq}\{\kappa_G(\sigma,r),\kappa_G(\tau,r)\}),\ \delta_G(\sigma,\tau)+\varepsilon\bigr)$

The ordering $\preceq$ is a fixed ordering of rooted simplex types; using it makes the simplex-pair component symmetric.

## Graph-level feature counts

Let $s\in\{0,1\}$ indicate whether self-pairs are included. For each root, consider unordered pairs with replacement from $\mathcal{C}_{q}(G;r)$ when $s=1$, and unordered distinct pairs when $s=0$. Represent an unordered pair by the equivalence class $[\sigma,\tau]$ under the relation $(\sigma,\tau)\sim(\tau,\sigma)$. The eligible pair collection is

$\mathcal{P}_s(G;r)=\{[\sigma,\tau]:\sigma,\tau\in\mathcal{C}_{q}(G;r),\ (\sigma\ne\tau)\lor(s=1)\}$

The count of a symbolic feature $\phi$ in graph $G$ is

$c_G(\phi)=\sum_{r\in V}\sum_{[\sigma,\tau]\in\mathcal{P}_{s}(G;r)}\mathbf{1}[L\leq\delta_G(\sigma,\tau)+\varepsilon\leq U]\,\mathbf{1}[\phi=\phi_G(r,\sigma,\tau)]$

The sum counts occurrences, including repeated occurrences of the same labeled pattern at different roots or in different parts of the graph. If a pair of cliques has multiple common vertices, each such vertex can serve as the root and contributes its own occurrence.

The uncompressed representation is the count function

$c_G:\mathcal{F}\rightarrow\mathbb{N}_0$

where $\mathcal{F}$ is the set of possible symbolic rooted simplex-pair types and $\mathbb{N}_0$ denotes the nonnegative integers. It can be viewed as a sparse vector because only finitely many features occur in any finite graph.

## Fixed-width vector representation

For learning, symbolic features are mapped to a fixed coordinate set $\{1,\ldots,m\}$, with $m=2^b$ for a chosen width $b$. Let $h:\mathcal{F}\rightarrow\{1,\ldots,m\}$ be the deterministic feature mapping. The coordinate value for graph $G$ is

$x_G(j)=\sum_{\phi\in\mathcal{F},\ h(\phi)=j}c_G(\phi),\quad j=1,\ldots,m$

If different symbolic features map to the same coordinate, their counts add. The resulting vector has nonnegative occurrence-count values and is typically sparse. It does not normalize counts or weight features by class.

## Protein contact graph interpretation

In the enzyme workflow, each graph vertex represents a C$\alpha$ atom from a selected protein chain, and an edge represents a residue pair whose spatial separation lies in the chosen contact window. The graph's vertices are labeled by an amino-acid category, and its edges receive contact labels (or optional distance-bin labels). The simplex-pair features are then generated from this labeled contact graph.

With the current notebook settings, the contact window is 4–8 Å, the node alphabet is `aa20`, the maximum simplex dimension is $q=3$, the representative distance interval is $[L,U]=[0,4]$, self-pairs are included, and the fixed vector width is $2^{11}$. The vectorizer constructor itself defaults to $2^{14}$ coordinates. The contact window determines which graph edges exist; the 0–4 feature distance limit instead counts shortest paths through those edges.

## Interpretation and scope

Each coordinate aggregates evidence for co-occurring, root-anchored labeled clique patterns at a selected representative distance. The representation therefore retains information about amino-acid labels, labeled local contact structure, and graph-scale separation, while replacing the original graph by counts of these patterns.

The feature vector is a summary rather than a lossless encoding of the graph. Different graphs can produce the same counts, and finite-width coordinate mapping can merge distinct symbolic patterns. The construction is also defined relative to its vertex-ordering convention because that convention selects simplex representatives and orders labels in rooted types. These properties should be considered when interpreting similarity or classifier results.
