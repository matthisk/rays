const std = @import("std");
const vector = @import("vector.zig");
const Interval = @import("interval.zig");
const Ray = @import("ray.zig").Ray;

const Vector3 = vector.Vector3;
const splat3 = vector.splat3;

// Axis-aligned bounding box.
const Aabb = @This();

x: Interval = Interval{},
y: Interval = Interval{},
z: Interval = Interval{},

pub fn init(a: Vector3, b: Vector3) Aabb {
    const x = Interval{ .min = @min(a[0], b[0]), .max = @max(a[0], b[0]) };
    const y = Interval{ .min = @min(a[1], b[1]), .max = @max(a[1], b[1]) };
    const z = Interval{ .min = @min(a[2], b[2]), .max = @max(a[2], b[2]) };

    return Aabb{ .x = x, .y = y, .z = z };
}

pub fn from(box1: Aabb, box2: Aabb) Aabb {
    return Aabb{
        .x = Interval.from(box1.x, box2.x),
        .y = Interval.from(box1.y, box2.y),
        .z = Interval.from(box1.z, box2.z),
    };
}

pub fn axis(self: Aabb, n: usize) Interval {
    return switch (n) {
        0 => self.x,
        1 => self.y,
        2 => self.z,
        else => @panic("illegal axis"),
    };
}

pub fn hit(self: Aabb, ray: Ray, ray_t: Interval) bool {
    var ray_t_min = ray_t.min;
    var ray_t_max = ray_t.max;

    for (0..3) |a| {
        const invD = 1 / ray.direction[a];
        const orig = ray.origin[a];

        var t0 = (self.axis(a).min - orig) * invD;
        var t1 = (self.axis(a).max - orig) * invD;

        if (invD < 0) {
            const tmp = t1;
            t1 = t0;
            t0 = tmp;
        }

        if (t0 > ray_t.min) ray_t_min = t0;
        if (t1 < ray_t.max) ray_t_max = t1;

        if (ray_t_max <= ray_t_min) {
            return false;
        }
    }

    return true;
}

test "hit" {
    const aabb = Aabb.init(Vector3{ 1, 1, 0 }, Vector3{ 6, 6, 1 });
    const ray = Ray.init(Vector3{ 0, 0, 0 }, Vector3{ 3, 5, 1 });

    try std.testing.expectEqual(true, aabb.hit(ray, Interval{ .min = -std.math.inf(f64), .max = std.math.inf(f64) }));
}

test "hit negative direction" {
    const aabb = Aabb.init(Vector3{ -1, -1, 0 }, Vector3{ -6, -6, -1 });
    const ray = Ray.init(Vector3{ 0, 0, 0 }, Vector3{ -3, -5, -1 });

    try std.testing.expectEqual(true, aabb.hit(ray, Interval{ .min = -std.math.inf(f64), .max = std.math.inf(f64) }));
}

test "no hit" {
    const aabb = Aabb.init(Vector3{ 1, 1, 1 }, Vector3{ 1, 1, 1 });
    const ray = Ray.init(Vector3{ 0, 0, 0 }, Vector3{ 1.5, 1.5, 0 });

    try std.testing.expectEqual(false, aabb.hit(ray, Interval{ .min = -std.math.inf(f64), .max = std.math.inf(f64) }));
}

test "no hit negative direction" {
    const aabb = Aabb.init(Vector3{ -1, -1, -1 }, Vector3{ -1, -1, -1 });
    const ray = Ray.init(Vector3{ 0, 0, 0 }, Vector3{ -1.5, -1.5, 0 });

    try std.testing.expectEqual(false, aabb.hit(ray, Interval{ .min = -std.math.inf(f64), .max = std.math.inf(f64) }));
}
