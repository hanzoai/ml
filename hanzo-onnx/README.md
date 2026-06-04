# hanzo-onnx

This crate adds ONNX support to hanzo

## FAQ

#### Missing protoc installation when compiling hanzo-onnx

The hanzo-onnx dependency prost-build no longer comes bundled with prost
binaries. This could cause the following error when attempting to compile
hanzo-onnx:

```
error: failed to run custom build command for `hanzo-onnx`
Caused by: // (...)
  Could not find `protoc` installation and this build crate cannot proceed without this knowledge.
```

To fix this issue install protoc on your system and make it available in your
system `PATH`. See the [protoc
documentation](https://grpc.io/docs/protoc-installation/) for more information.
