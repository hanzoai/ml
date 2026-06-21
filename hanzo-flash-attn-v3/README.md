# hanzo-flash-attn-v3

FlashAttention-3 (Hopper / `sm_90a`) attention layer for the Hanzo ML framework.

## Status: incomplete WIP — excluded from the workspace build

This crate is **excluded** from the workspace (`exclude` in the root `Cargo.toml`)
and does not build as-is. Only the **head-dim 512** sm90 kernels were ever written
(`hkernel/flash_fwd_hdim512_{fp16,bf16}[_gqaN]_sm90.cu`, 12 kernels + `flash_api.cu`).

`src/lib.rs` advertises head dims 128/256/512 and `flash_api.cu`'s `HEADDIM_SWITCH`
dispatches across 64/128/256/512, but the **hdim 64/128/256 kernel sources were never
created** (confirmed absent in git history and on the evo/spark GPU fleet; no
`generate_kernels.py`). `build.rs` `KERNEL_FILES` has been trimmed to the 13 sources
that exist.

To make it a workspace member, either:
1. narrow `lib.rs` validation + `flash_api.cu` `HEADDIM_SWITCH` to hdim 512 only and
   build-verify on a Hopper (sm90) GPU, or
2. author the missing hdim 64/128/256 sm90 kernels.

Preserved here (debranded from the orphaned upstream `hanzo-flash-attn-v3`) so the
Hopper hdim512 work is not lost. Requires CUDA + `compute_cap >= 90` (`cudaforge` build).
