const std = @import("std");
const vecs = @import("vec3.zig");
const rays = @import("ray.zig");

test "ray at with t = 1" {
    const origin = vecs.Vec3.init(10, 20, 30);
    const direction = vecs.Vec3.init(10, 10, 30);

    const ray = &rays.Ray.init(
        origin,
        direction,
    );

    const result = ray.at(1);

    try std.testing.expect(result.x == 20);
    try std.testing.expect(result.y == 30);
    try std.testing.expect(result.z == 60);
}

test "ray at with t = 2" {
    const origin = vecs.Vec3.init(10, 20, 30);
    const direction = vecs.Vec3.init(10, 10, 30);

    const ray = &rays.Ray.init(
        origin,
        direction,
    );

    const result = ray.at(2);

    try std.testing.expect(result.x == 30);
    try std.testing.expect(result.y == 40);
    try std.testing.expect(result.z == 90);
}
