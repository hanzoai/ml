# hanzo-3d

3D representations, geometry and file I/O for image-to-3D pipelines. Not a model: the pieces a
3D model decodes into and exports from.

- **`representations`**: `Mesh`, `GaussianSplat`, `Voxel`, and `Slat`, the structured latent of
  the TRELLIS family (TRELLIS, TripoSR, Pixal3D).
- **`geometry`**: `Vec3`, `Mat3`, `Quat`, and a pinhole `Camera` whose `project` and
  `backproject` are exact inverses.
- **`io`**: OBJ and PLY meshes, and 3D Gaussian Splatting PLY; every writer has a reader that
  round-trips it exactly.

No dependencies beyond `std`, so the crate and its round-trip tests build anywhere.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
