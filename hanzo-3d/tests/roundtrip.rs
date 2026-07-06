//! The correctness gate: every writer's output parses back to the exact value, and the camera
//! projection is an exact inverse. If these pass, the crate is a faithful serialization foundation.

use hanzo_3d::geometry::{Camera, Quat, Vec3};
use hanzo_3d::io::{
    mesh_to_obj, mesh_to_ply, obj_to_mesh, ply_to_mesh, ply_to_splat, splat_to_ply,
};
use hanzo_3d::representations::{Gaussian, GaussianSplat};
use hanzo_3d::unit_cube;

#[test]
fn mesh_obj_roundtrip_is_exact() {
    let cube = unit_cube();
    let back = obj_to_mesh(&mesh_to_obj(&cube)).expect("parse OBJ");
    assert_eq!(back.vertices, cube.vertices, "OBJ vertices differ");
    assert_eq!(back.faces, cube.faces, "OBJ faces differ");
    println!(
        "[hanzo-3d] OBJ roundtrip: {} verts / {} faces bit-match",
        back.vertex_count(),
        back.face_count()
    );
}

#[test]
fn mesh_ply_roundtrip_is_exact() {
    let cube = unit_cube();
    let back = ply_to_mesh(&mesh_to_ply(&cube)).expect("parse PLY");
    assert_eq!(back.vertices, cube.vertices, "PLY vertices differ");
    assert_eq!(back.faces, cube.faces, "PLY faces differ");
    println!(
        "[hanzo-3d] PLY mesh roundtrip: {} verts / {} faces bit-match",
        back.vertex_count(),
        back.face_count()
    );
}

#[test]
fn gaussian_splat_ply_roundtrip_is_exact() {
    let g = GaussianSplat {
        splats: vec![
            Gaussian {
                position: Vec3::new(0.1, -0.2, 0.3),
                scale: Vec3::new(0.01, 0.02, 0.03),
                rotation: Quat::new(0.7071068, 0.7071068, 0.0, 0.0),
                color: [0.9, 0.4, 0.1],
                opacity: 0.75,
            },
            Gaussian {
                position: Vec3::new(1.0, 2.0, 3.0),
                scale: Vec3::new(0.5, 0.5, 0.5),
                rotation: Quat::IDENTITY,
                color: [0.0, 1.0, 0.0],
                opacity: 1.0,
            },
        ],
    };
    let back = ply_to_splat(&splat_to_ply(&g)).expect("parse splat PLY");
    assert_eq!(back, g, "splat PLY roundtrip differs");
    println!("[hanzo-3d] 3DGS PLY roundtrip: {} splats bit-match", back.len());
}

#[test]
fn camera_project_backproject_is_identity() {
    let cam = Camera::look_forward(500.0, 500.0, 320.0, 240.0);
    let mut max_err = 0.0f32;
    for &p in &[
        Vec3::new(0.1, 0.2, 2.0),
        Vec3::new(-0.5, 0.3, 5.0),
        Vec3::new(1.0, -1.0, 3.5),
    ] {
        let (u, v, d) = cam.project(p);
        let back = cam.backproject(u, v, d);
        let e = (back - p).length();
        max_err = max_err.max(e);
    }
    assert!(max_err < 1e-4, "camera roundtrip error {max_err} too large");
    println!("[hanzo-3d] camera project->backproject identity: max_err {max_err:.2e}");
}

#[test]
fn empty_mesh_roundtrips() {
    let empty = hanzo_3d::Mesh::default();
    assert_eq!(ply_to_mesh(&mesh_to_ply(&empty)).unwrap(), empty);
    assert_eq!(obj_to_mesh(&mesh_to_obj(&empty)).unwrap(), empty);
    println!("[hanzo-3d] empty-mesh roundtrip ok");
}
