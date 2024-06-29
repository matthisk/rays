const std = @import("std");
const vector = @import("vector.zig");
const Color = @import("color.zig").Color;
const RtwImage = @import("image.zig").RtwImage;
const Interval = @import("interval.zig");
const Perlin = @import("perlin.zig");

const Vector3 = vector.Vector3;

pub const Texture = union(enum) {
    solid_color: SolidColor,
    checker: CheckerTexture,
    image: ImageTexture,
    noise: NoiseTexture,
    marble: MarbleTexture,
    wood: WoodTexture,

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

pub const ImageTexture = struct {
    image: *RtwImage,

    pub fn init(image: *RtwImage) Texture {
        const imageTexture = ImageTexture{
            .image = image,
        };

        return Texture{ .image = imageTexture };
    }

    pub fn value(self: ImageTexture, u: f64, v: f64, p: Vector3) Color {
        _ = p;
        if (self.image.height() <= 0) return Vector3{ 0, 0, 0 };

        const u_n = (Interval{ .min = 0, .max = 1 }).clamp(u);
        const v_n = 1.0 - (Interval{ .min = 0, .max = 1 }).clamp(v);

        const i = u_n * @as(f64, @floatFromInt(self.image.width())) - 1;
        const j = v_n * @as(f64, @floatFromInt(self.image.height())) - 1;

        const pixel = self.image.pixelData(@intFromFloat(i), @intFromFloat(j));

        const color_scale = 1.0 / 255.0;
        return Vector3{ color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2] };
    }
};

pub const NoiseTexture = struct {
    perlin: *Perlin,
    scale: f64,

    pub fn init(perlin: *Perlin, scale: f64) Texture {
        return Texture{ .noise = NoiseTexture{ .perlin = perlin, .scale = scale } };
    }

    pub fn value(self: NoiseTexture, u: f64, v: f64, p: Vector3) Color {
        _ = v;
        _ = u;
        //return Color{ 1, 1, 1 } * vector.splat3(0.5) * vector.splat3(1.0 + self.perlin.noise(vector.splat3(self.scale) * p));
        // return Color{ 1, 1, 1 } * vector.splat3(0.5) * vector.splat3(1.0 + self.perlin.noise(vector.splat3(self.scale) * Vector3{ u, v, 0 }));
        return Color{ 1, 1, 1 } * vector.splat3(self.perlin.turbulence(p, 3));
    }
};

pub const MarbleTexture = struct {
    perlin: *Perlin,
    power: f64 = 10,

    pub fn init(perlin: *Perlin) Texture {
        return Texture{ .marble = MarbleTexture{ .perlin = perlin } };
    }

    pub fn value(self: MarbleTexture, u: f64, v: f64, p: Vector3) Color {
        _ = v;
        _ = u;
        return Color{ 0.5, 0.2, 0.2 } * vector.splat3(1 + @sin(self.power * p[0] + 10 * self.perlin.turbulence(p, 7)));
    }
};

pub const WoodTexture = struct {
    period: f64 = 100, // period determines how many lines are carved in the wood.
    turb_power: f64 = 0.2,
    perlin: *Perlin,

    pub fn init(perlin: *Perlin) Texture {
        return Texture{ .wood = WoodTexture{ .perlin = perlin } };
    }

    pub fn value(self: WoodTexture, u: f64, v: f64, p: Vector3) Color {
        _ = p;
        const x_value = (Interval{ .min = 0, .max = 1 }).clamp(u);
        const y_value = 1.0 - (Interval{ .min = 0, .max = 1 }).clamp(v);

        const dist = @sqrt(x_value * x_value + y_value * y_value) + self.turb_power * self.perlin.turbulence(vector.Vector3{ x_value, y_value, 0 }, 10);
        const sin_value = @fabs(@sin(self.period * dist * 3.14));

        return Color{ 0.9, 0.6, 0.5 } * vector.splat3(sin_value);
    }
};

test "NoiseTexture" {
    const perlin = try Perlin.init(std.testing.allocator);
    defer perlin.destroy(std.testing.allocator);

    const texture = NoiseTexture.init(perlin);
    const color = texture.value(0, 0, Vector3{ 0, 0, 0 });
    _ = color;
}
