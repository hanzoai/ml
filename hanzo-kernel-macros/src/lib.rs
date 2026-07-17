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
//!
//! # Intrinsic islands
//!
//! Inside a `#[kernel]` body, `island! { ... }` is a scoped, target-gated region: one inner slice of
//! the kernel picks a target-specific instruction/idiom while the kernel keeps ONE signature, ONE
//! launch, ONE CPU oracle. It is not a real Rust macro — cubecl's `#[cube]` rejects unknown body
//! macros — so `#[kernel]` rewrites it here, before delegating to the engine, into a **comptime
//! `match`** over the kernel's `Target` parameter. A comptime match lowers only the selected arm, so
//! each backend compiles exactly its own arm and nothing else.
//!
//! ```ignore
//! let dp = island! {
//!     // accelerated arm: a target-specific instruction/idiom
//!     cuda | rocm | metal | vulkan => { w.dot(x) }
//!     // NORMATIVE fallback — CPU-oracle semantics, REQUIRED. Every unlisted target takes this.
//!     default => { w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] }
//! };
//! ```
//!
//! An island sits in either position: bound (`let x = island! { ... };`) when the arms yield a value,
//! or bare (`island! { ... };`) when they act on shared memory — a warp-cooperative tile op has no
//! per-lane result to bind, so the bare form is the natural one for it.
//!
//! The scrutinee is found BY TYPE: the kernel must take one `#[comptime] <name>: Target` parameter
//! (any name), which the launch layer fills from the runtime — so the branch site names no target and
//! the selection stays out of the authoring surface. `default` is mandatory and defines the oracle:
//! the CPU runtime is always tagged `Target::Cpu`, which no accelerated arm claims, so CPU runs
//! `default`.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    visit_mut::{self, VisitMut},
    Expr, ExprMacro, FnArg, Ident, ItemFn, Pat, Stmt, Token, Type,
};

/// The backends a kernel can name. Each maps 1:1 to a runtime the same source lowers to.
const TARGETS: &[&str] = &["cpu", "cuda", "rocm", "metal", "vulkan", "webgpu"];

/// The island surface target names, in the fixed order the exhaustive `match` is generated in, paired
/// with their `Target` enum variant. This is the single source of truth for the island backend set.
const ISLAND_TARGETS: &[(&str, &str)] = &[
    ("cpu", "Cpu"),
    ("cuda", "Cuda"),
    ("rocm", "Rocm"),
    ("metal", "Metal"),
    ("vulkan", "Vulkan"),
    ("webgpu", "WebGpu"),
];

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

/// `#[kernel(targets(...))]`: rewrite any intrinsic `island!`s, validate the backend list, then lower
/// the function via the engine.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as Args);
    let mut func = parse_macro_input!(item as ItemFn);

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

    // Lower `island! { ... }` regions to a comptime `match` over the kernel's `Target` param BEFORE
    // the engine attribute runs (cubecl rejects unknown body macros, so this must happen here).
    let mut rewrite = IslandRewrite { target: find_target_param(&func), error: None };
    rewrite.visit_item_fn_mut(&mut func);
    if let Some(err) = rewrite.error {
        return err.to_compile_error().into();
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

// ================================================================================================
// Intrinsic islands: the `island!` rewrite (see the module docs).
// ================================================================================================

/// Visitor that replaces `island! { ... }` expressions with a comptime `match <target> { ... }`.
struct IslandRewrite {
    /// The kernel's `Target` parameter identifier (the match scrutinee), found by type.
    target: Option<Ident>,
    /// First error encountered (a `VisitMut` cannot return one).
    error: Option<syn::Error>,
}

impl VisitMut for IslandRewrite {
    /// `island! { ... }` at STATEMENT position — the form an island takes when its arms act on shared
    /// memory instead of yielding a value (a warp-cooperative tile op has no per-lane result to bind).
    /// syn parses a braced macro statement as `Stmt::Macro`, never `Stmt::Expr`, so `visit_expr_mut`
    /// alone would never see it. Re-seat it as the equivalent expression statement and rewrite that,
    /// preserving the original trailing semicolon (or its absence, when the island is the block's value).
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        if let Stmt::Macro(sm) = stmt {
            if sm.mac.path.is_ident("island") {
                let mut expr = Expr::Macro(ExprMacro {
                    attrs: sm.attrs.clone(),
                    mac: sm.mac.clone(),
                });
                let semi = sm.semi_token;
                self.visit_expr_mut(&mut expr);
                *stmt = Stmt::Expr(expr, semi);
                return;
            }
        }
        visit_mut::visit_stmt_mut(self, stmt);
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Macro(mac) = expr {
            if mac.mac.path.is_ident("island") {
                match &self.target {
                    Some(target) => match expand_island(target, mac.mac.tokens.clone()) {
                        Ok(replacement) => *expr = replacement,
                        Err(err) => {
                            self.error.get_or_insert(err);
                        }
                    },
                    None => {
                        self.error.get_or_insert(syn::Error::new_spanned(
                            &mac.mac,
                            "a kernel using `island!` must take a `#[comptime] <name>: Target` parameter",
                        ));
                    }
                }
                return;
            }
        }
        visit_mut::visit_expr_mut(self, expr);
    }
}

/// Find the kernel's `Target` parameter by type — the island scrutinee. Binding by type (not by a
/// magic name) lets the author name it anything and keeps the branch site free of plumbing.
fn find_target_param(func: &ItemFn) -> Option<Ident> {
    func.sig.inputs.iter().find_map(|arg| match arg {
        FnArg::Typed(pt) if type_is_target(&pt.ty) => match &*pt.pat {
            Pat::Ident(pi) => Some(pi.ident.clone()),
            _ => None,
        },
        _ => None,
    })
}

/// Whether a type is (path-syntactically) `Target` — the island tag.
fn type_is_target(ty: &Type) -> bool {
    matches!(ty, Type::Path(tp) if tp.path.segments.last().is_some_and(|s| s.ident == "Target"))
}

/// One island selector: `default`, or one-or-more target names joined by `|`.
enum Selector {
    Default,
    Targets(Vec<Ident>),
}

/// One island arm: a selector and its body expression.
struct IslandArm {
    selector: Selector,
    body: Expr,
}

/// The parsed interior of `island! { ... }`.
struct IslandInput {
    arms: Vec<IslandArm>,
}

impl Parse for IslandInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut arms = Vec::new();
        while !input.is_empty() {
            let first: Ident = input.parse()?;
            let selector = if first == "default" {
                Selector::Default
            } else {
                let mut targets = vec![first];
                while input.peek(Token![|]) {
                    let _: Token![|] = input.parse()?;
                    targets.push(input.parse()?);
                }
                Selector::Targets(targets)
            };
            let _: Token![=>] = input.parse()?;
            let body: Expr = input.parse()?;
            arms.push(IslandArm { selector, body });
            // Arms may be comma-separated; the comma is optional (also after a `{ }` block body).
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }
        Ok(IslandInput { arms })
    }
}

/// Compile `island! { ... }` to an exhaustive comptime `match <target> { ... }`: every `Target`
/// variant maps to its listed arm's body, and every unlisted variant to the mandatory `default`.
fn expand_island(target: &Ident, tokens: TokenStream2) -> syn::Result<Expr> {
    let input: IslandInput = syn::parse2(tokens)?;

    let mut default_body: Option<Expr> = None;
    let mut bodies: Vec<(&'static str, Expr)> = Vec::new();

    for arm in input.arms {
        match arm.selector {
            Selector::Default => {
                if default_body.is_some() {
                    return Err(syn::Error::new(Span::call_site(), "island! has more than one `default` arm"));
                }
                default_body = Some(arm.body);
            }
            Selector::Targets(targets) => {
                for t in targets {
                    let variant = ISLAND_TARGETS
                        .iter()
                        .find(|(name, _)| t == name)
                        .map(|(_, variant)| *variant)
                        .ok_or_else(|| {
                            syn::Error::new(t.span(), format!("unknown island target `{t}`; valid: {}", island_target_names()))
                        })?;
                    if bodies.iter().any(|(v, _)| *v == variant) {
                        return Err(syn::Error::new(t.span(), format!("island target `{t}` is listed twice")));
                    }
                    bodies.push((variant, arm.body.clone()));
                }
            }
        }
    }

    let default_body = default_body.ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            "island! requires a `default => { ... }` arm — the normative fallback that defines the CPU-oracle semantics",
        )
    })?;

    // Exhaustive: one arm per Target variant, in fixed order, so there is no wildcard and the CPU
    // variant provably resolves to `default`.
    let match_arms = ISLAND_TARGETS.iter().map(|(_, variant)| {
        let v = Ident::new(variant, Span::call_site());
        let body = bodies.iter().find(|(name, _)| name == variant).map(|(_, b)| b).unwrap_or(&default_body);
        quote! { Target::#v => #body }
    });

    Ok(parse_quote! { match #target { #(#match_arms),* } })
}

fn island_target_names() -> String {
    ISLAND_TARGETS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
}
