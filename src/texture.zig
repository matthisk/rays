const std = @import("std");
const vector = @import("vector.zig");
const Color = @import("color.zig").Color;
const _ = @import("image.zig");

const Vector3 = vector.Vector3;

pub const Texture = union(enum) {
    solid_color: SolidColor,
    checker: CheckerTexture,

    pub fn value(self: Texture, u: f64, v: f64, p: Vector3) Color {
        return switch (self) {
            inline else => |case| case.value(u, v, p),
        };
    }
};

pub const SolidColor = struct {
    color: Color,

    pub fn init(color: Color) Texture {
        return Texture{ .solid_color = SolidColor{
            .color = color,
        } };
    }

    pub fn value(self: SolidColor, u: f64, v: f64, p: Vector3) Color {
        _ = p;
        _ = v;
        _ = u;
        return self.color;
    }
};

pub const CheckerTexture = struct {
    inv_scale: f64,
    even: SolidColor,
    odd: SolidColor,

    pub fn init(scale: f64, even: SolidColor, odd: SolidColor) Texture {
        return Texture{ .checker = CheckerTexture{
            .inv_scale = 1.0 / scale,
            .even = even,
            .odd = odd,
        } };
    }

    pub fn initWithColors(scale: f64, even: Color, odd: Color) Texture {
        return Texture{ .checker = CheckerTexture{
            .inv_scale = 1.0 / scale,
            .even = SolidColor{
                .color = even,
            },
            .odd = SolidColor{
                .color = odd,
            },
        } };
    }

    pub fn value(self: CheckerTexture, u: f64, v: f64, p: Vector3) Color {
        const xInt: usize = @intFromFloat(@fabs(@floor(self.inv_scale * p[0])));
        const yInt: usize = @intFromFloat(@fabs(@floor(self.inv_scale * p[1])));
        const zInt: usize = @intFromFloat(@fabs(@floor(self.inv_scale * p[2])));

        const is_even = (xInt + yInt + zInt) % 2 == 0;

        return if (is_even) self.even.value(u, v, p) else self.odd.value(u, v, p);
    }
};

test "checker texture" {
    const even = Color{ 0, 0, 0 };
    const odd = Color{ 0, 0, 0 };
    const texture = CheckerTexture.initWithColors(1, even, odd);

    const color_1 = texture.value(0, 0, Vector3{ 0, 0, 0 });
    const color_2 = texture.value(0, 0, Vector3{ 1, 1, 1 });
    const color_3 = texture.value(0, 0, Vector3{ 1, 0, 1 });
    const color_4 = texture.value(0, 0, Vector3{ 0, 1, 0 });

    try std.testing.expectEqual(even, color_1);
    try std.testing.expectEqual(odd, color_2);
    try std.testing.expectEqual(even, color_3);
    try std.testing.expectEqual(odd, color_4);
}
