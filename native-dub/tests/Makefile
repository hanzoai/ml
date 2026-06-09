# native-dub e2e test suite
#
#   make e2e         full suite (CI + components + full pipeline) on the GPU box (spark)
#   make ci          structural cargo tests only (no GPU)
#   make components  per-component GPU regression
#   make full        full-pipeline e2e
#   make charmatch   build the pure-Rust char-match helper

SHELL := /usr/bin/env bash
HERE  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: e2e ci components full charmatch

e2e:
	"$(HERE)run_e2e_tests.sh" all

ci:
	"$(HERE)run_e2e_tests.sh" ci

components:
	"$(HERE)run_e2e_tests.sh" components

full:
	"$(HERE)run_e2e_tests.sh" full

charmatch:
	cd "$(HERE)charmatch" && cargo build --release
