//! Serialization for the 3D representations — the glTF-adjacent formats a client consumes.
//!
//! Hand-rolled, std-only, ASCII: [`Mesh`] ⇄ OBJ and PLY, [`GaussianSplat`] ⇄ the standard
//! 3D-Gaussian-Splatting PLY layout. Every writer has a matching reader, and the roundtrip is
//! exact (the gate in `tests/`). Binary PLY and full glTF are deliberately out of the base crate —
//! ASCII keeps the invariant "what we wrote, we read back bit-for-bit" trivially checkable.

use crate::geometry::{Quat, Vec3};
use crate::representations::{Gaussian, GaussianSplat, Mesh};
use std::fmt::Write as _;

/// An I/O error — a parse failure with the offending context.
#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hanzo-3d io: {}", self.0)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

// ---- OBJ (mesh) -----------------------------------------------------------------------------

/// Write a [`Mesh`] as Wavefront OBJ (`v` + optional `vn` + 1-indexed `f`).
pub fn mesh_to_obj(m: &Mesh) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# hanzo-3d OBJ");
    for v in &m.vertices {
        let _ = writeln!(s, "v {} {} {}", v.x, v.y, v.z);
    }
    for n in &m.normals {
        let _ = writeln!(s, "vn {} {} {}", n.x, n.y, n.z);
    }
    let has_n = !m.normals.is_empty();
    for f in &m.faces {
        let (a, b, c) = (f[0] + 1, f[1] + 1, f[2] + 1);
        if has_n {
            let _ = writeln!(s, "f {a}//{a} {b}//{b} {c}//{c}");
        } else {
            let _ = writeln!(s, "f {a} {b} {c}");
        }
    }
    s
}

/// Parse a Wavefront OBJ into a [`Mesh`]. Handles `v`, `vn`, and `f` with `a`, `a/b`, or `a//c`
/// index forms; ignores everything else.
pub fn obj_to_mesh(src: &str) -> Result<Mesh> {
    let mut m = Mesh::default();
    for (lineno, line) in src.lines().enumerate() {
        let line = line.trim();
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let v = read_vec3(&mut it).map_err(|e| Error(format!("line {}: v: {}", lineno + 1, e.0)))?;
                m.vertices.push(v);
            }
            Some("vn") => {
                let n = read_vec3(&mut it).map_err(|e| Error(format!("line {}: vn: {}", lineno + 1, e.0)))?;
                m.normals.push(n);
            }
            Some("f") => {
                let mut idx = [0u32; 3];
                for (k, slot) in idx.iter_mut().enumerate() {
                    let tok = it.next().ok_or_else(|| Error(format!("line {}: f needs 3 verts", lineno + 1)))?;
                    // "a", "a/b", "a//c" — the vertex index is the first field, 1-based.
                    let first = tok.split('/').next().unwrap_or(tok);
                    let one_based: u32 = first
                        .parse()
                        .map_err(|_| Error(format!("line {}: bad face index {:?}", lineno + 1, tok)))?;
                    if one_based == 0 {
                        return err(format!("line {}: face vertex 0 (OBJ is 1-indexed)", lineno + 1));
                    }
                    let _ = k;
                    *slot = one_based - 1;
                }
                m.faces.push(idx);
            }
            _ => {}
        }
    }
    Ok(m)
}

// ---- PLY (mesh) -----------------------------------------------------------------------------

/// Write a [`Mesh`] as ASCII PLY (vertex `x y z` + `element face` with `vertex_indices`).
pub fn mesh_to_ply(m: &Mesh) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "ply");
    let _ = writeln!(s, "format ascii 1.0");
    let _ = writeln!(s, "comment hanzo-3d");
    let _ = writeln!(s, "element vertex {}", m.vertices.len());
    let _ = writeln!(s, "property float x");
    let _ = writeln!(s, "property float y");
    let _ = writeln!(s, "property float z");
    let _ = writeln!(s, "element face {}", m.faces.len());
    let _ = writeln!(s, "property list uchar int vertex_indices");
    let _ = writeln!(s, "end_header");
    for v in &m.vertices {
        let _ = writeln!(s, "{} {} {}", v.x, v.y, v.z);
    }
    for f in &m.faces {
        let _ = writeln!(s, "3 {} {} {}", f[0], f[1], f[2]);
    }
    s
}

/// Parse an ASCII PLY mesh (the layout [`mesh_to_ply`] emits).
pub fn ply_to_mesh(src: &str) -> Result<Mesh> {
    let (header, body) = split_ply_header(src)?;
    let n_vert = header.count("vertex")?;
    let n_face = header.count("face")?;
    let mut lines = body.lines().filter(|l| !l.trim().is_empty());
    let mut m = Mesh::default();
    for _ in 0..n_vert {
        let line = lines.next().ok_or_else(|| Error("ply: truncated vertices".into()))?;
        let mut it = line.split_whitespace();
        m.vertices.push(read_vec3(&mut it)?);
    }
    for _ in 0..n_face {
        let line = lines.next().ok_or_else(|| Error("ply: truncated faces".into()))?;
        let mut it = line.split_whitespace();
        let k: usize = it.next().and_then(|t| t.parse().ok()).ok_or_else(|| Error("ply: bad face count".into()))?;
        if k != 3 {
            return err(format!("ply: only triangles supported, got a {k}-gon"));
        }
        let a = parse_u32(&mut it, "face")?;
        let b = parse_u32(&mut it, "face")?;
        let c = parse_u32(&mut it, "face")?;
        m.faces.push([a, b, c]);
    }
    Ok(m)
}

// ---- PLY (gaussian splat) -------------------------------------------------------------------

/// The per-splat properties, in the order the standard 3DGS PLY writes them.
const SPLAT_PROPS: &[&str] = &[
    "x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0",
    "f_dc_1", "f_dc_2", "opacity",
];

/// Write a [`GaussianSplat`] as the standard 3D-Gaussian-Splatting ASCII PLY.
pub fn splat_to_ply(g: &GaussianSplat) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "ply");
    let _ = writeln!(s, "format ascii 1.0");
    let _ = writeln!(s, "comment hanzo-3d gaussian-splat");
    let _ = writeln!(s, "element vertex {}", g.splats.len());
    for p in SPLAT_PROPS {
        let _ = writeln!(s, "property float {p}");
    }
    let _ = writeln!(s, "end_header");
    for sp in &g.splats {
        let _ = writeln!(
            s,
            "{} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            sp.position.x,
            sp.position.y,
            sp.position.z,
            sp.scale.x,
            sp.scale.y,
            sp.scale.z,
            sp.rotation.w,
            sp.rotation.x,
            sp.rotation.y,
            sp.rotation.z,
            sp.color[0],
            sp.color[1],
            sp.color[2],
            sp.opacity,
        );
    }
    s
}

/// Parse a 3DGS ASCII PLY into a [`GaussianSplat`] (the layout [`splat_to_ply`] emits).
pub fn ply_to_splat(src: &str) -> Result<GaussianSplat> {
    let (header, body) = split_ply_header(src)?;
    let n = header.count("vertex")?;
    let mut out = GaussianSplat::default();
    let mut lines = body.lines().filter(|l| !l.trim().is_empty());
    for _ in 0..n {
        let line = lines.next().ok_or_else(|| Error("ply: truncated splats".into()))?;
        let mut it = line.split_whitespace();
        let mut f = [0f32; 14];
        for (i, slot) in f.iter_mut().enumerate() {
            *slot = it
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| Error(format!("ply: splat missing property {}", SPLAT_PROPS[i])))?;
        }
        out.splats.push(Gaussian {
            position: Vec3::new(f[0], f[1], f[2]),
            scale: Vec3::new(f[3], f[4], f[5]),
            rotation: Quat::new(f[6], f[7], f[8], f[9]),
            color: [f[10], f[11], f[12]],
            opacity: f[13],
        });
    }
    Ok(out)
}

// ---- shared parsing helpers -----------------------------------------------------------------

/// The header text of a PLY, with element counts pulled out.
struct PlyHeader<'a> {
    text: &'a str,
}

impl PlyHeader<'_> {
    /// The count declared by `element <name> <n>`.
    fn count(&self, name: &str) -> Result<usize> {
        for line in self.text.lines() {
            let mut it = line.split_whitespace();
            if it.next() == Some("element") && it.next() == Some(name) {
                return it
                    .next()
                    .and_then(|t| t.parse().ok())
                    .ok_or_else(|| Error(format!("ply: bad count for element {name}")));
            }
        }
        // A missing element is simply zero of them.
        Ok(0)
    }
}

fn split_ply_header(src: &str) -> Result<(PlyHeader<'_>, &str)> {
    if !src.trim_start().starts_with("ply") {
        return err("not a PLY (missing 'ply' magic)");
    }
    let marker = "end_header";
    let pos = src.find(marker).ok_or_else(|| Error("ply: no end_header".into()))?;
    let header = &src[..pos];
    let body = &src[pos + marker.len()..];
    Ok((PlyHeader { text: header }, body))
}

fn read_vec3<'a>(it: &mut impl Iterator<Item = &'a str>) -> Result<Vec3> {
    let x = parse_f32(it, "vec3.x")?;
    let y = parse_f32(it, "vec3.y")?;
    let z = parse_f32(it, "vec3.z")?;
    Ok(Vec3::new(x, y, z))
}

fn parse_f32<'a>(it: &mut impl Iterator<Item = &'a str>, what: &str) -> Result<f32> {
    it.next()
        .and_then(|t| t.parse().ok())
        .ok_or_else(|| Error(format!("expected f32 for {what}")))
}

fn parse_u32<'a>(it: &mut impl Iterator<Item = &'a str>, what: &str) -> Result<u32> {
    it.next()
        .and_then(|t| t.parse().ok())
        .ok_or_else(|| Error(format!("expected u32 for {what}")))
}
