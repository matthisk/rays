const std = @import("std");
const rand = @import("rand.zig");

pub const Vector3 = @Vector(3, f64);
pub const Vector4 = @Vector(4, f64);

pub fn dot(u: Vector3, v: Vector3) f64 {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

pub fn cross(u: Vector3, v: Vector3) Vector3 {
    return Vector3{
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    };
}

pub fn length(u: Vector3) f64 {
    return std.math.sqrt(lengthSquared(u));
}

pub fn lengthSquared(u: Vector3) f64 {
    return u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
}

pub fn nearZero(u: Vector3) bool {
    return u[0] < 1e-8 and u[1] < 1e-8 and u[2] < 1e-8;
}

pub fn splat3(v: f64) Vector3 {
    return @as(Vector3, @splat(v));
}

pub fn splat4(v: f64) Vector4 {
    return @as(Vector4, @splat(v));
}

pub fn reflect(v: Vector3, n: Vector3) Vector3 {
    return v - n * splat3(dot(v, n) * 2);
}

pub fn refract(uv: Vector3, n: Vector3, etai_overetat: f64) Vector3 {
    const cos_theta = @min(dot(-uv, n), 1.0);
    const r_out_perp = (uv + n * splat3(cos_theta)) * splat3(etai_overetat);
    const r_out_parallel = n * splat3(-std.math.sqrt(std.math.fabs(1.0 - lengthSquared(r_out_perp))));

    return r_out_perp + r_out_parallel;
}

pub fn unitVector(v: Vector3) Vector3 {
    return v / splat3(length(v));
}

pub fn randomUnitVector() Vector3 {
    return unitVector(randomInUnitSphere());
}

pub fn randomOnHemisphere(normal: Vector3) Vector3 {
    const on_unit_sphere = randomUnitVector();
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

pub fn randomInUnitSphere() Vector3 {
    while (true) {
        const p = randomBetween(-1, 1);
        if (lengthSquared(p) < 1) {
            return p;
        }
    }
}

pub fn randomInUnitDisk() Vector3 {
    while (true) {
        const p = Vector3{ rand.randomBetween(-1, 1), rand.randomBetween(-1, 1), 0 };
        if (lengthSquared(p) < 1) {
            return p;
        }
    }
}

pub fn random() Vector3 {
    return Vector3{ rand.randomFloat(), rand.randomFloat(), rand.randomFloat() };
}

pub fn randomBetween(min: f64, max: f64) Vector3 {
    return Vector3{ rand.randomBetween(min, max), rand.randomBetween(min, max), rand.randomBetween(min, max) };
}

test "u . v" {
    const u = Vector3{ 10, 20, 30 };
    const v = Vector3{ 10, 20, 30 };

    const result = dot(u, v);

    try std.testing.expectEqual(@as(f64, 1400), result);
}

test "u x v" {
    const u = Vector3{ 10, 20, 30 };
    const v = Vector3{ 30, 20, 10 };

    const result = cross(u, v);

    try std.testing.expectEqual(Vector3{ -400, 800, -400 }, result);
}

test "vector3 length" {
    const u = Vector3{ 10, 20, 30 };

    try std.testing.expectApproxEqAbs(@as(f64, 37.4165738677), length(u), 0.0001);
}

test "reflect" {
    const u = Vector3{ 10, 20, 30 };
    const n = Vector3{ 0, 1, 0 };

    const result = reflect(u, n);

    try std.testing.expectEqual(Vector3{ 10, -20, 30 }, result);
}

test "refract" {
    const uv = Vector3{ 10, 20, 30 };
    const n = Vector3{ 0, 1, 0 };

    const result = refract(uv, n, 1);

    try std.testing.expectApproxEqAbs(@as(f64, 10), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, -3.160696e1), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 30), result[2], 0.001);
}
