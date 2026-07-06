//! Minimal 3D geometry — `Vec3`, `Mat3`, `Quat`, and a pinhole `Camera`.
//!
//! Pure `f32`, no dependency. The one non-trivial thing here is the camera: Pixal3D and the
//! TRELLIS family condition on *pixel-aligned* features, which means projecting world points to
//! the image plane and back. `project` and `backproject` are exact inverses (up to depth), which
//! the roundtrip test pins.

/// A 3-vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Vec3 {
        let l = self.length();
        if l == 0.0 {
            self
        } else {
            self * (1.0 / l)
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

/// A row-major 3×3 matrix — a rotation, or the camera's world→camera basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    /// Rows: `[r0, r1, r2]`, each a `Vec3`.
    pub r: [Vec3; 3],
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3 {
        r: [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
    };

    /// Matrix · vector.
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3::new(self.r[0].dot(v), self.r[1].dot(v), self.r[2].dot(v))
    }

    /// Transpose — for a rotation matrix this is the inverse.
    pub fn transpose(self) -> Mat3 {
        Mat3 {
            r: [
                Vec3::new(self.r[0].x, self.r[1].x, self.r[2].x),
                Vec3::new(self.r[0].y, self.r[1].y, self.r[2].y),
                Vec3::new(self.r[0].z, self.r[1].z, self.r[2].z),
            ],
        }
    }
}

/// A unit quaternion `(w, x, y, z)` — the rotation carried by a Gaussian splat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quat {
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Quat { w, x, y, z }
    }

    /// The 3×3 rotation this quaternion represents.
    pub fn to_mat3(self) -> Mat3 {
        let Quat { w, x, y, z } = self;
        Mat3 {
            r: [
                Vec3::new(
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ),
                Vec3::new(
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ),
                Vec3::new(
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ),
            ],
        }
    }
}

/// A pinhole camera: intrinsics `(fx, fy, cx, cy)` and a world→camera rigid transform `(rot, t)`.
///
/// Camera space looks down `+z`; a world point `p` maps to camera space as `rot·p + t`, then to
/// a pixel `(u, v)` by the usual `u = fx·x/z + cx`, `v = fy·y/z + cy`. `backproject` inverts it
/// given the depth `z` that `project` returns.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub rot: Mat3,
    pub t: Vec3,
}

impl Camera {
    /// A camera at the origin looking down `+z` with the given intrinsics.
    pub fn look_forward(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Camera { fx, fy, cx, cy, rot: Mat3::IDENTITY, t: Vec3::ZERO }
    }

    /// World point → `(u, v, depth)`. `depth` is the camera-space `z`.
    pub fn project(&self, world: Vec3) -> (f32, f32, f32) {
        let c = self.rot.mul_vec(world) + self.t;
        let u = self.fx * c.x / c.z + self.cx;
        let v = self.fy * c.y / c.z + self.cy;
        (u, v, c.z)
    }

    /// `(u, v, depth)` → world point. Exact inverse of [`project`](Camera::project).
    pub fn backproject(&self, u: f32, v: f32, depth: f32) -> Vec3 {
        let cx = (u - self.cx) / self.fx * depth;
        let cy = (v - self.cy) / self.fy * depth;
        let cam = Vec3::new(cx, cy, depth);
        // world = rotᵀ · (cam − t)
        self.rot.transpose().mul_vec(cam - self.t)
    }
}
