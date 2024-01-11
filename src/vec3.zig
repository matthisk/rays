const std = @import("std");
const rand = @import("rand.zig");

pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn zero() Vec3 {
        return Vec3{ .x = 0, .y = 0, .z = 0 };
    }

    pub fn init(x: f32, y: f32, z: f32) Vec3 {
        return Vec3{ .x = x, .y = y, .z = z };
    }

    pub fn plus(self: Vec3, v: Vec3) Vec3 {
        return init(self.x + v.x, self.y + v.y, self.z + v.z);
    }

    pub fn minus(self: Vec3, v: Vec3) Vec3 {
        return init(self.x - v.x, self.y - v.y, self.z - v.z);
    }

    pub fn multiply(self: Vec3, t: f32) Vec3 {
        return init(self.x * t, self.y * t, self.z * t);
    }

    pub fn multiplyByVec3(self: Vec3, v: Vec3) Vec3 {
        return init(self.x * v.x, self.y * v.y, self.z * v.z);
    }

    pub fn divide(self: Vec3, d: f32) Vec3 {
        return self.multiply(1 / d);
    }

    pub fn dot(self: Vec3, v: Vec3) f32 {
        return self.x * v.x + self.y * v.y + self.z * v.z;
    }

    pub fn length(self: Vec3) f32 {
        return std.math.sqrt(self.length_squared());
    }

    pub fn length_squared(self: Vec3) f32 {
        return self.x * self.x + self.y * self.y + self.z * self.z;
    }
};

pub fn unit_vector(v: Vec3) Vec3 {
    return v.divide(v.length());
}

pub fn random_unit_vector() Vec3 {
    return unit_vector(random_in_unit_sphere());
}

pub fn random_on_hemisphere(normal: Vec3) Vec3 {
    const on_unit_sphere = random_unit_vector();
    if (on_unit_sphere.dot(normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return on_unit_sphere.multiply(-1);
    }
}

pub fn random_in_unit_sphere() Vec3 {
    while (true) {
        const p = random_between(-1, 1);
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

pub fn random() Vec3 {
    return Vec3.init(rand.random_float(), rand.random_float(), rand.random_float());
}

pub fn random_between(min: f32, max: f32) Vec3 {
    return Vec3.init(rand.random_between(min, max), rand.random_between(min, max), rand.random_between(min, max));
}
