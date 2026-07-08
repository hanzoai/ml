//! # hanzo-3d
//!
//! The 3D foundation for Hanzo's generative-3D stack — the representations, geometry, and I/O that
//! any image-to-3D model in the TRELLIS family (Pixal3D, TRELLIS, TripoSR) decodes into and exports.
//!
//! It is deliberately *not* a model: the diffusion backbone reuses `hanzo-engine`'s existing Flux /
//! Qwen-Image pipeline. What had no home in the stack — and what this crate is — are the three
//! orthogonal pieces every 3D pipeline needs and none of them ship:
//!
//! - [`representations`] — [`Mesh`](representations::Mesh),
//!   [`GaussianSplat`](representations::GaussianSplat), [`Voxel`](representations::Voxel), and the
//!   TRELLIS structured latent [`Slat`](representations::Slat).
//! - [`geometry`] — [`Vec3`](geometry::Vec3) / [`Mat3`](geometry::Mat3) / [`Quat`](geometry::Quat)
//!   and a pinhole [`Camera`](geometry::Camera) whose `project`/`backproject` are exact inverses
//!   (the pixel-aligned conditioning Pixal3D needs).
//! - [`io`] — OBJ and PLY for meshes, and the standard 3D-Gaussian-Splatting PLY for splats, every
//!   writer paired with an exact-roundtrip reader.
//!
//! Zero external dependencies: the crate compiles and its roundtrip gates run with nothing but
//! `std`, so it is a stable base the model crates build on rather than a dependency-fetch risk.

pub mod geometry;
pub mod io;
pub mod representations;

pub use geometry::{Camera, Mat3, Quat, Vec3};
pub use io::Error as IoError;
pub use representations::{Gaussian, GaussianSplat, Material, Mesh, Slat, Voxel};

/// A unit cube mesh (8 vertices, 12 triangles) — a fixture for the roundtrip gates and a handy
/// smoke-test object for downstream pipelines.
pub fn unit_cube() -> Mesh {
    let v = |x, y, z| Vec3::new(x, y, z);
    let vertices = vec![
        v(0.0, 0.0, 0.0),
        v(1.0, 0.0, 0.0),
        v(1.0, 1.0, 0.0),
        v(0.0, 1.0, 0.0),
        v(0.0, 0.0, 1.0),
        v(1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0),
        v(0.0, 1.0, 1.0),
    ];
    let faces = vec![
        [0, 1, 2],
        [0, 2, 3], // back
        [4, 6, 5],
        [4, 7, 6], // front
        [0, 4, 5],
        [0, 5, 1], // bottom
        [3, 2, 6],
        [3, 6, 7], // top
        [0, 3, 7],
        [0, 7, 4], // left
        [1, 5, 6],
        [1, 6, 2], // right
    ];
    Mesh::new(vertices, faces)
}
