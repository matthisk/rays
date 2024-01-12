const vecs = @import("vec3.zig");
const rand = @import("rand.zig");

pub const Color = vecs.Vec3;

pub fn toBgra(color: Color) u32 {
    const r: u32 = @intFromFloat(color.x * 255.999);
    const g: u32 = @intFromFloat(color.y * 255.999);
    const b: u32 = @intFromFloat(color.z * 255.999);

    return 255 << 24 | r << 16 | g << 8 | b;
}
