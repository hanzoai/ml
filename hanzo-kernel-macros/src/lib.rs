//! `#[kernel(targets(...))]` — the hanzo-kernel authoring attribute.
//!
//! `kernel` names what the thing is (a GPU kernel); `targets(...)` names where it runs. One Rust
//! function, lowered to the listed backends. Replaces the upstream engine's brand attribute so kernel
//! source reads in Hanzo's own vocabulary and nothing else.
//!
//! ```ignore
//! #[kernel(targets(cuda, metal, vulkan, webgpu, cpu))]
//! fn ntt_mul<F: Field>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>, #[comptime] n: usize) { ... }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Ident, ItemFn, Token,
};

/// The backends a kernel can name. Each maps 1:1 to a runtime the same source lowers to.
const TARGETS: &[&str] = &["cpu", "cuda", "rocm", "metal", "vulkan", "webgpu"];

/// Parsed `kernel(targets(a, b, ...), [unchecked])` arguments.
struct Args {
    targets: Vec<Ident>,
    unchecked: bool,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut targets = Vec::new();
        let mut unchecked = false;
        // Comma-separated top-level items: `targets(...)` and optionally `unchecked`.
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            if key == "targets" {
                let content;
                syn::parenthesized!(content in input);
                let list: Punctuated<Ident, Token![,]> = content.parse_terminated(Ident::parse, Token![,])?;
                targets.extend(list);
            } else if key == "unchecked" {
                unchecked = true;
            } else {
                return Err(syn::Error::new(key.span(), "expected `targets(...)` or `unchecked`"));
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }
        Ok(Args { targets, unchecked })
    }
}

/// `#[kernel(targets(...))]`: validate the backend list, then lower the function via the engine.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as Args);
    let func = parse_macro_input!(item as ItemFn);

    if args.targets.is_empty() {
        return syn::Error::new_spanned(&func.sig.ident, "a kernel must name at least one target, e.g. `#[kernel(targets(cuda, metal, vulkan, webgpu, cpu))]`")
            .to_compile_error()
            .into();
    }
    for t in &args.targets {
        if !TARGETS.iter().any(|k| t == k) {
            return syn::Error::new(t.span(), format!("unknown target `{t}`; valid targets: {}", TARGETS.join(", ")))
                .to_compile_error()
                .into();
        }
    }

    // Same expansion the engine attribute produces; `targets(...)` is a checked, self-documenting
    // declaration of the backend set (a future codegen hook can gate per target from this list).
    let inner = if args.unchecked {
        quote! { #[cube(launch_unchecked)] #func }
    } else {
        quote! { #[cube(launch)] #func }
    };
    inner.into()
}

/// `#[device]` — an on-GPU helper function callable from a `#[kernel]`, inlined at the call site.
/// A kernel is *launched*; a device function is not an entry point, it is a piece of a kernel.
#[proc_macro_attribute]
pub fn device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    quote! { #[cube] #func }.into()
}
