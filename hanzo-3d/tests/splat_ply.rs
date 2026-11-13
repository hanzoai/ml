//! Check the file values independently of the crate's own roundtrip.

use hanzo_3d::io::{ply_to_splat, splat_to_ply};
use hanzo_3d::{Gaussian, GaussianSplat, Quat, Vec3};
use std::f32::consts::LN_2;

// Written by hand: these fixtures must not use splat_to_ply to build their input.
const HEADER: &str = "ply\n\
format ascii 1.0\n\
element vertex 1\n\
property float x\n\
property float y\n\
property float z\n\
property float scale_0\n\
property float scale_1\n\
property float scale_2\n\
property float rot_0\n\
property float rot_1\n\
property float rot_2\n\
property float rot_3\n\
property float f_dc_0\n\
property float f_dc_1\n\
property float f_dc_2\n\
property float opacity\n\
end_header\n";

fn gaussian(scale: Vec3, opacity: f32) -> Gaussian {
    Gaussian {
        position: Vec3::new(1.0, 2.0, 3.0),
        scale,
        rotation: Quat::IDENTITY,
        color: [0.2, 0.4, 0.6],
        opacity,
    }
}

fn exported_values(splat: Gaussian) -> Vec<f32> {
    let ply = splat_to_ply(&GaussianSplat {
        splats: vec![splat],
    });
    ply.split_once("end_header\n")
        .expect("PLY header")
        .1
        .split_whitespace()
        .map(|value| value.parse().expect("float property"))
        .collect()
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2e-6 * expected.abs().max(1.0),
        "expected {expected}, got {actual}"
    );
}

#[test]
fn export_writes_log_scales_and_logit_opacity() {
    for (scale, opacity, expected_scale, expected_opacity) in [
        (Vec3::new(1.0, 2.0, 3.0), 0.5, [0.0, LN_2, 1.0986123], 0.0),
        (
            Vec3::new(0.5, 1.0, 2.0),
            0.25,
            [-LN_2, 0.0, LN_2],
            -1.0986123,
        ),
        (
            Vec3::new(0.01, 0.02, 0.03),
            0.75,
            [-4.6051702, -3.912023, -3.506558],
            1.0986123,
        ),
    ] {
        let fields = exported_values(gaussian(scale, opacity));
        assert_eq!(fields.len(), 14);
        assert_eq!(&fields[..3], &[1.0, 2.0, 3.0]);
        for (actual, expected) in fields[3..6].iter().zip(expected_scale) {
            assert_close(*actual, expected);
        }
        assert_eq!(&fields[6..13], &[1.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6]);
        assert_close(fields[13], expected_opacity);
    }
}

#[test]
fn import_decodes_a_handwritten_log_encoded_splat() {
    let ply = format!("{HEADER}1 2 3 -0.6931472 0 0.6931472 1 0 0 0 0.2 0.4 0.6 -1.0986123\n");
    let parsed = ply_to_splat(&ply).expect("parse fixture");
    assert_eq!(parsed.len(), 1);
    let splat = parsed.splats[0];
    assert_eq!(splat.position, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(splat.rotation, Quat::IDENTITY);
    assert_eq!(splat.color, [0.2, 0.4, 0.6]);
    assert_close(splat.scale.x, 0.5);
    assert_close(splat.scale.y, 1.0);
    assert_close(splat.scale.z, 2.0);
    assert_close(splat.opacity, 0.25);
}

#[test]
fn opacity_endpoints_use_infinite_logits() {
    for (opacity, logit) in [(0.0, f32::NEG_INFINITY), (1.0, f32::INFINITY)] {
        let fields = exported_values(gaussian(Vec3::new(1.0, 1.0, 1.0), opacity));
        assert_eq!(fields[13], logit);
        let ply = format!("{HEADER}1 2 3 0 0 0 1 0 0 0 0.2 0.4 0.6 {logit}\n");
        assert_eq!(ply_to_splat(&ply).unwrap().splats[0].opacity, opacity);
    }
}

#[test]
fn very_small_opacity_does_not_underflow_during_import() {
    let ply = format!("{HEADER}1 2 3 0 0 0 1 0 0 0 0.2 0.4 0.6 -100\n");
    let opacity = ply_to_splat(&ply).unwrap().splats[0].opacity;
    // sigmoid(-100), allowing one f32 subnormal step for rounding.
    assert!((f64::from(opacity) - 3.720075976020836e-44).abs() <= f64::from(f32::from_bits(1)));
}

#[test]
fn very_small_opacity_survives_export_and_import() {
    let source = GaussianSplat {
        splats: vec![gaussian(Vec3::new(1.0, 1.0, 1.0), 1e-40)],
    };
    let back = ply_to_splat(&splat_to_ply(&source)).unwrap();
    assert!((back.splats[0].opacity / source.splats[0].opacity - 1.0).abs() < 1e-4);
}

#[test]
fn zero_scale_is_encoded_as_negative_infinity() {
    let fields = exported_values(gaussian(Vec3::new(0.0, 0.25, 1.0), 0.5));
    assert_eq!(fields[3], f32::NEG_INFINITY);
    assert_close(fields[4], -1.3862944);
    assert_eq!(fields[5], 0.0);
    let ply = format!("{HEADER}1 2 3 -inf -1.3862944 0 1 0 0 0 0.2 0.4 0.6 0\n");
    let splat = ply_to_splat(&ply).unwrap().splats[0];
    assert_eq!(splat.scale.x, 0.0);
    assert_close(splat.scale.y, 0.25);
    assert_eq!(splat.scale.z, 1.0);
}

#[test]
fn empty_splat_roundtrips() {
    let empty = GaussianSplat::default();
    assert_eq!(ply_to_splat(&splat_to_ply(&empty)).unwrap(), empty);
}
