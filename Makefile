.PHONY: clean-ptx clean test

clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > hanzo-ml-kernels/src/lib.rs
	touch hanzo-ml-kernels/build.rs
	touch hanzo-ml-examples/build.rs
	touch hanzo-ml-flash-attn/build.rs

clean:
	cargo clean

test:
	cargo test

all: test
