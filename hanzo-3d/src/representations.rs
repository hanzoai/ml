//! The 3D value types the generative-3D family produces.
//!
//! One crate, four representations — the axes a Pixal3D / TRELLIS pipeline moves along:
//! a sparse [`Voxel`] occupancy grid (the coarse structure stage), a [`Slat`] structured latent
//! (per-occupied-voxel feature — TRELLIS's unified encoding), and the two decoded outputs a
//! client actually renders: a [`Mesh`] (surface + PBR) and a [`GaussianSplat`] cloud (radiance).
//!
//! These are plain data. I/O lives in [`crate::io`]; math in [`crate::geometry`].

use crate::geometry::{Quat, Vec3};

/// A triangle mesh with optional normals, UVs, and scalar PBR material.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    /// Triangle vertex indices, `[a, b, c]` into `vertices`.
    pub faces: Vec<[u32; 3]>,
    /// Per-vertex normals (empty if absent).
    pub normals: Vec<Vec3>,
    /// Per-vertex `(u, v)` texture coordinates (empty if absent).
    pub uvs: Vec<[f32; 2]>,
    /// Scalar PBR (a full texture stack is out of scope for the base type).
    pub material: Material,
}

/// Scalar PBR material — the base-color / roughness / metallic a decoded mesh carries before a
/// texture bake.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    pub base_color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
}

impl Default for Material {
    fn default() -> Self {
        Material { base_color: [0.8, 0.8, 0.8, 1.0], roughness: 1.0, metallic: 0.0 }
    }
}

impl Mesh {
    pub fn new(vertices: Vec<Vec3>, faces: Vec<[u32; 3]>) -> Self {
        Mesh { vertices, faces, ..Default::default() }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }
}

/// One 3D Gaussian: position, anisotropic scale, orientation, view-independent color, opacity.
///
/// This is the per-splat record of the standard 3D-Gaussian-Splatting PLY, minus higher-order SH
/// (only the DC color term is kept — the base type is view-independent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gaussian {
    pub position: Vec3,
    /// Per-axis scale (log-space in the file format; linear here).
    pub scale: Vec3,
    pub rotation: Quat,
    /// DC (view-independent) RGB.
    pub color: [f32; 3],
    pub opacity: f32,
}

/// A radiance field as a cloud of [`Gaussian`]s.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GaussianSplat {
    pub splats: Vec<Gaussian>,
}

impl GaussianSplat {
    pub fn len(&self) -> usize {
        self.splats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.splats.is_empty()
    }
}

/// A sparse occupancy grid at a cubic `resolution` — the coarse-structure stage output.
///
/// Only occupied cell coordinates are stored; `resolution` is the side length of the implicit
/// `res³` lattice.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Voxel {
    pub resolution: u32,
    pub occupied: Vec<[u32; 3]>,
}

impl Voxel {
    pub fn new(resolution: u32) -> Self {
        Voxel { resolution, occupied: Vec::new() }
    }
}

/// A structured latent (TRELLIS "SLAT"): a feature vector attached to each occupied voxel.
///
/// `coords[i]` is a lattice coordinate and `feats[i]` its latent — the sparse conditioning the
/// shape/texture flow decodes from. `coords.len() == feats.len()` is the invariant.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Slat {
    pub resolution: u32,
    pub coords: Vec<[u32; 3]>,
    pub feats: Vec<Vec<f32>>,
}

impl Slat {
    /// The per-voxel feature width (0 if empty). Assumes a uniform width, which the constructor
    /// preserves.
    pub fn feature_dim(&self) -> usize {
        self.feats.first().map_or(0, |f| f.len())
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}
