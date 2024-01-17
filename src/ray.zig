const std = @import("std");
const vector = @import("vector.zig");

const Vector3 = vector.Vector3;

pub const Ray = struct {
    origin: Vector3,
    direction: Vector3,

    pub fn init(origin: Vector3, direction: Vector3) Ray {
        return Ray{ .origin = origin, .direction = direction };
    }

    pub fn at(self: Ray, t: f64) Vector3 {
        return self.origin + self.direction * vector.splat3(t);
    }
};

test "ray at with t = 1" {
    const origin = Vector3{ 10, 20, 30 };
    const direction = Vector3{ 10, 10, 30 };

    const ray = &Ray.init(
        origin,
        direction,
    );

    const result = ray.at(1);

    try std.testing.expect(result[0] == 20);
    try std.testing.expect(result[1] == 30);
    try std.testing.expect(result[2] == 60);
}

test "ray at with t = 2" {
    const origin = Vector3{ 10, 20, 30 };
    const direction = Vector3{ 10, 10, 30 };

    const ray = &Ray.init(
        origin,
        direction,
    );

    const result = ray.at(2);

    try std.testing.expect(result[0] == 30);
    try std.testing.expect(result[1] == 40);
    try std.testing.expect(result[2] == 90);
}
