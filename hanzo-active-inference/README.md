# Finite-state active inference

Canonical CPU mathematical core for local world factors. No tensor backend, model
download, network service or actuator is needed. Tables use A[action][outcome][state]
and B[action][next][current]. All columns are stochastic; C is a strictly positive
preference distribution. Structural safety constraints use hard policy masks.

Implemented:

- Variational free energy F(Q) = E_Q[log Q(s) - log P(o,s)]. Exact Bayesian
  filtering minimizes it, giving F* = -log P(o). Prediction is a separate B step.
- G(pi) sums KL(Q(o)|C) + E_Q(s) H[P(o|s)] over a finite open-loop horizon.
  Equivalently G = -E log C(o) - I(s;o). Both decompositions are returned.
- Q(pi) = softmax(log E(pi) - gamma G(pi)), with stable normalization,
  nonnegative precision and host-owned hard admissibility masks.
- Dirichlet A/B expected-count learning from the exact adjacent-state joint
  posterior. Learning is immutable, validated, and never learns preferences.

A local factor is exact. A large factored/loopy graph is not made exact by using
this primitive: scheduling, message convergence and separator consistency belong
to the host. Explicit policies are bounded (256 candidates, horizon 16); there is
no hidden exponential enumeration. Per factor state/outcome limits are 256.
Zero-evidence observations, malformed probabilities and all-masked policies fail.

Not implemented: contingent closed-loop policy trees, expected information gain
about parameters, continuous generalized coordinates, learned policy precision,
GPU tensor kernels, or global variational message passing. The Gaussian Sekai
smoke model is a separate approximate latent-dynamics experiment, not a proof of
this categorical model or full physical-world competence.

References: [discrete-state synthesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/),
[pymdp control](https://pymdp-rtd.readthedocs.io/en/latest/api/control/),
[Whence EFE](https://doi.org/10.1162/neco_a_01354).

```sh
cargo test -p hanzo-active-inference
cargo build -p hanzo-active-inference --release
```

The binary reads one model/inference request per JSON line and returns one result
or error per line. It is a local development boundary. Engine hosts call the Rust
library directly; they do not spawn the binary on each request.
