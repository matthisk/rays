const std = @import("std");
const vecs = @import("vec3.zig");
const rand = @import("rand.zig");
const Interval = @import("interval.zig");

pub const Color = vecs.Vec3;

pub const ColorAndSamples = struct {
    color: Color,
    number_of_samples: u64,
};

pub fn toBgra(color: Color) u32 {
    const r: u32 = @intFromFloat(color.x * 255.999);
    const g: u32 = @intFromFloat(color.y * 255.999);
    const b: u32 = @intFromFloat(color.z * 255.999);

    return 255 << 24 | r << 16 | g << 8 | b;
}

pub fn toGamma(color: Color, scale: f64) Color {
    var r = color.x;
    var g = color.y;
    var b = color.z;

    r *= scale;
    g *= scale;
    b *= scale;

    r = linearToGamma(r);
    g = linearToGamma(g);
    b = linearToGamma(b);

    const intensity = Interval{ .min = 0.0, .max = 0.999 };

    return Color.init(intensity.clamp(r), intensity.clamp(g), intensity.clamp(b));
}

pub fn linearToGamma(linear_component: f64) f64 {
    return std.math.sqrt(linear_component);
}
