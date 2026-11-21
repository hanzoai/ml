# Hanzo ML developer targets. `make` or `make help` lists them.
#
# The kernel crates (hanzo-kernel, hanzo-*-kernels, hanzo-flash-attn) sit outside the
# workspace because their build scripts declare `links=`; they are reached through
# --manifest-path. GPU backends are cargo features, e.g.
#   make example NAME=quantized FEATURES=cuda

CARGO  ?= cargo
MDBOOK ?= mdbook

.DEFAULT_GOAL := help
.PHONY: help check build test test-kernel fmt fmt-check clippy example book book-serve \
	publish-order clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

check: ## Type-check every workspace crate (the CI gate)
	$(CARGO) check --workspace

build: ## Build every workspace crate for the CPU
	$(CARGO) build --workspace

test: ## Run the workspace tests on the CPU
	$(CARGO) test --workspace

test-kernel: ## Test the kernel DSL against its CPU oracle
	$(CARGO) test --manifest-path hanzo-kernel/Cargo.toml

fmt: ## Format the workspace and its path crates
	$(CARGO) fmt --all

fmt-check: ## Fail if formatting differs
	$(CARGO) fmt --all -- --check

clippy: ## Lint with warnings as errors, as CI does
	$(CARGO) clippy --workspace --tests --examples --benches -- -D warnings

example: ## Run an example: NAME=<example> [FEATURES=cuda|metal|mkl|...]
	@test -n "$(NAME)" || { echo "usage: make example NAME=<example> [FEATURES=<features>]"; exit 2; }
	$(CARGO) run --release -p hanzo-ml-examples --example $(NAME) $(if $(FEATURES),--features $(FEATURES))

book: ## Build the book into hanzo-ml-book/book
	$(MDBOOK) build hanzo-ml-book

book-serve: ## Serve the book on localhost:3000 and rebuild on change
	$(MDBOOK) serve --open hanzo-ml-book

publish-order: ## Print the publishable crates in dependency order
	scripts/publish-order

clean: ## Remove build output
	$(CARGO) clean
