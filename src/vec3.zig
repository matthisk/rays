const std = @import("std");
const rand = @import("rand.zig");

pub const Vec3 = struct {
    x: f64,
    y: f64,
    z: f64,

    pub fn zero() Vec3 {
        return Vec3{ .x = 0, .y = 0, .z = 0 };
    }

    pub fn init(x: f64, y: f64, z: f64) Vec3 {
        return Vec3{ .x = x, .y = y, .z = z };
    }

    pub fn plus(self: Vec3, v: Vec3) Vec3 {
        return init(self.x + v.x, self.y + v.y, self.z + v.z);
    }

    pub fn minus(self: Vec3, v: Vec3) Vec3 {
        return init(self.x - v.x, self.y - v.y, self.z - v.z);
    }

    pub fn multiply(self: Vec3, t: f64) Vec3 {
        return init(self.x * t, self.y * t, self.z * t);
    }

    pub fn multiplyByVec3(self: Vec3, v: Vec3) Vec3 {
        return init(self.x * v.x, self.y * v.y, self.z * v.z);
    }

    pub fn divide(self: Vec3, d: f64) Vec3 {
        return self.multiply(1 / d);
    }

    pub fn dot(self: Vec3, v: Vec3) f64 {
        return self.x * v.x + self.y * v.y + self.z * v.z;
    }

    pub fn cross(self: Vec3, v: Vec3) Vec3 {
        return Vec3.init(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        );
    }

    pub fn length(self: Vec3) f64 {
        return std.math.sqrt(self.lengthSquared());
    }

    pub fn lengthSquared(self: Vec3) f64 {
        return self.x * self.x + self.y * self.y + self.z * self.z;
    }

    pub fn nearZero(self: Vec3) bool {
        const s = 1e-8;
        return (std.math.fabs(self.x) < s) and (std.math.fabs(self.y) < s) and (std.math.fabs(self.z) < s);
    }
};

pub fn reflect(v: Vec3, n: Vec3) Vec3 {
    return v.minus(n.multiply(v.dot(n) * 2));
}

pub fn refract(uv: Vec3, n: Vec3, etai_overetat: f64) Vec3 {
    const cos_theta = @min(uv.multiply(-1).dot(n), 1.0);
    const r_out_perp = uv.plus(n.multiply(cos_theta)).multiply(etai_overetat);
    const r_out_parallel = n.multiply(-std.math.sqrt(std.math.fabs(1.0 - r_out_perp.lengthSquared())));

    return r_out_perp.plus(r_out_parallel);
}

pub fn unitVector(v: Vec3) Vec3 {
    return v.divide(v.length());
}

pub fn randomUnitVector() Vec3 {
    return unitVector(randomInUnitSphere());
}

pub fn randomOnHemisphere(normal: Vec3) Vec3 {
    const on_unit_sphere = randomUnitVector();
    if (on_unit_sphere.dot(normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return on_unit_sphere.multiply(-1);
    }
}

pub fn randomInUnitSphere() Vec3 {
    while (true) {
        const p = randomBetween(-1, 1);
        if (p.lengthSquared() < 1) {
            return p;
        }
    }
}

pub fn randomInUnitDisk() Vec3 {
    while (true) {
        const p = Vec3.init(rand.randomBetween(-1, 1), rand.randomBetween(-1, 1), 0);
        if (p.lengthSquared() < 1) {
            return p;
        }
    }
}

pub fn random() Vec3 {
    return Vec3.init(rand.randomFloat(), rand.randomFloat(), rand.randomFloat());
}

pub fn randomBetween(min: f64, max: f64) Vec3 {
    return Vec3.init(rand.randomBetween(min, max), rand.randomBetween(min, max), rand.randomBetween(min, max));
}

test "empty vec3" {
    const vec = Vec3.zero();

    try std.testing.expect(vec.x == 0 and vec.y == 0 and vec.z == 0);
}

test "init vec3" {
    const vec = Vec3.init(10, 20, 30);

    try std.testing.expect(vec.x == 10 and vec.y == 20 and vec.z == 30);
}

test "plus vec3" {
    const vec1 = Vec3.init(10, 20, 30);
    const vec2 = Vec3.init(10, 20, 30);

    const result = vec1.plus(vec2);

    try std.testing.expect(result.x == 20 and result.y == 40 and result.z == 60);
}

test "multiply vec3 by vec3" {
    const vec1 = Vec3.init(10, 20, 30);
    const vec2 = Vec3.init(1, 2, 3);

    const result = vec1.multiplyByVec3(vec2);

    try std.testing.expect(result.x == 10 and result.y == 40 and result.z == 90);
}

test "multiply vec3 by f32" {
    const vec1 = Vec3.init(10, 20, 30);

    const result = vec1.multiply(2);

    try std.testing.expect(result.x == 20 and result.y == 40 and result.z == 60);
}

test "divide vec3" {
    const vec1 = Vec3.init(10, 20, 30);

    const result = vec1.divide(2);

    try std.testing.expect(result.x == 5 and result.y == 10 and result.z == 15);
}

test "dot vec3" {
    const vec1 = Vec3.init(10, 20, 30);
    const vec2 = Vec3.init(10, 20, 30);

    const result = vec1.dot(vec2);

    try std.testing.expectEqual(@as(f64, 1400), result);
}

test "length vec3" {
    const vec1 = Vec3.init(10, 20, 30);

    try std.testing.expectApproxEqAbs(@as(f64, 37.4165738677), vec1.length(), 0.0001);
}
