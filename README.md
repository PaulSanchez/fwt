Walsh transforms are useful in a variety of applications, such as image or
speech processing, filtering, and efficiently creating [very large statistical
designs of experiments](https://core.ac.uk/download/pdf/36728443.pdf).

Walsh functions are a binary-valued (±1) alternative to the more widely known
Fourier functions. Their time index can be represented using different orderings,
but regardless of the ordering used they constitute a complete orthogonal basis
for a vector space. Fast Walsh Transforms (FWTs)&mdash;similar to the well-known
Fast Fourier Transform&mdash;provide computationally efficient and numerically
stable calculations of the transform. This package provides FWT implementations
for sequency and Hadamard ordering. Both algorithms have O(*n* log(*n*)) time
complexity, where *n* is the length of the slice to be transformed and must
be a power of 2.
