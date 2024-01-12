const std = @import("std");
const vecs = @import("vec3.zig");

test "empty vec3" {
    const vec = vecs.Vec3.zero();

    try std.testing.expect(vec.x == 0 and vec.y == 0 and vec.z == 0);
}

test "init vec3" {
    const vec = vecs.Vec3.init(10, 20, 30);

    try std.testing.expect(vec.x == 10 and vec.y == 20 and vec.z == 30);
}

test "plus vec3" {
    const vec1 = vecs.Vec3.init(10, 20, 30);
    const vec2 = vecs.Vec3.init(10, 20, 30);

    const result = vec1.plus(vec2);

    try std.testing.expect(result.x == 20 and result.y == 40 and result.z == 60);
}

test "multiply vec3 by vec3" {
    const vec1 = vecs.Vec3.init(10, 20, 30);
    const vec2 = vecs.Vec3.init(1, 2, 3);

    const result = vec1.multiplyByVec3(vec2);

    try std.testing.expect(result.x == 10 and result.y == 40 and result.z == 90);
}

test "multiply vec3 by f32" {
    const vec1 = vecs.Vec3.init(10, 20, 30);

    const result = vec1.multiply(2);

    try std.testing.expect(result.x == 20 and result.y == 40 and result.z == 60);
}

test "divide vec3" {
    const vec1 = vecs.Vec3.init(10, 20, 30);

    const result = vec1.divide(2);

    try std.testing.expect(result.x == 5 and result.y == 10 and result.z == 15);
}

test "dot vec3" {
    const vec1 = vecs.Vec3.init(10, 20, 30);
    const vec2 = vecs.Vec3.init(10, 20, 30);

    const result = vec1.dot(vec2);

    try std.testing.expectEqual(@as(f32, 1400), result);
}

test "length vec3" {
    const vec1 = vecs.Vec3.init(10, 20, 30);

    try std.testing.expect(vec1.length() == 37.4165738677);
}
