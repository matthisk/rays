const std = @import("std");
const vector = @import("vector.zig");
const rand = @import("rand.zig");
const Interval = @import("interval.zig");

pub const Color = vector.Vector3;
pub const ColorAndSamples = vector.Vector4;

pub fn toBgra(color: Color) u32 {
    const r: u32 = @intFromFloat(color[0] * 255.999);
    const g: u32 = @intFromFloat(color[1] * 255.999);
    const b: u32 = @intFromFloat(color[2] * 255.999);

    return 255 << 24 | r << 16 | g << 8 | b;
}

pub fn toGamma(color: vector.Vector4) Color {
    const scale = 1.0 / color[3];

    var r = color[0];
    var g = color[1];
    var b = color[2];

    r *= scale;
    g *= scale;
    b *= scale;

    r = linearToGamma(r);
    g = linearToGamma(g);
    b = linearToGamma(b);

    const intensity = Interval{ .min = 0.0, .max = 0.999 };

    return Color{ intensity.clamp(r), intensity.clamp(g), intensity.clamp(b) };
}

pub fn linearToGamma(linear_component: f64) f64 {
    return std.math.sqrt(linear_component);
}
