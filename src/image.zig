const std = @import("std");
const c = @import("c.zig");
const absolutePath = @import("util.zig").absolutePath;
const Color = @import("color.zig").Color;

const ImageReadError = error{FailedToReadImage};

pub const RtwImage = struct {
    image_width: c_int = 0,
    image_height: c_int = 0,
    bytes_per_pixel: c_int = 0,
    bytes_per_scanline: c_int = undefined,
    raw: [*c]f32 = undefined,
    image_data: []f32 = undefined,

    pub fn init(allocator: std.mem.Allocator) !*RtwImage {
        var result = try allocator.create(RtwImage);
        result.* = RtwImage{};

        return result;
    }

    pub fn load(self: *RtwImage, filename: [:0]const u8) ImageReadError!void {
        self.raw = c.stbi_loadf(filename.ptr, &self.image_width, &self.image_height, &self.bytes_per_pixel, self.bytes_per_pixel);

        if (self.raw == null) {
            std.debug.print("failed to load image {s}\n", .{c.stbi_failure_reason()});
            return ImageReadError.FailedToReadImage;
        }

        self.bytes_per_scanline = self.image_width * self.bytes_per_pixel;
        // Convert c pointer to a zig slice.
        self.image_data = self.raw[0..@as(usize, @intCast(self.bytes_per_scanline * self.image_height))];
    }

    pub fn destroy(self: *RtwImage, allocator: std.mem.Allocator) void {
        c.stbi_image_free(self.raw);
        allocator.destroy(self);
    }

    pub fn pixelData(self: *RtwImage, x: usize, y: usize) Color {
        if (self.image_data.len == 0) return Color{ 255, 255, 255 };

        const bytes_per_scanline: usize = @intCast(self.bytes_per_scanline);
        const bytes_per_pixel: usize = @intCast(self.bytes_per_pixel);

        const start = y * bytes_per_scanline + x * bytes_per_pixel;
        const end = start + bytes_per_pixel;

        return convertToColor(self.image_data[start..end]);
    }

    pub fn width(self: *RtwImage) usize {
        return @intCast(self.image_width);
    }

    pub fn height(self: *RtwImage) usize {
        return @intCast(self.image_height);
    }
};

fn convertToColor(pixels: []f32) Color {
    return Color{ floatToByte(pixels[0]), floatToByte(pixels[1]), floatToByte(pixels[2]) };
}

fn floatToByte(f: f32) f64 {
    if (f < 0) return 0.0;
    if (1.0 < f) return 255;
    return f * 256.0;
}

test "RtwImage width and height" {
    const path = try absolutePath(std.testing.allocator, "assets/image.png");
    defer std.testing.allocator.free(path);

    var result = try RtwImage.init(std.testing.allocator);
    defer result.destroy(std.testing.allocator);

    try result.load(path);

    const width: usize = 408;
    const height: usize = 378;

    try std.testing.expectEqual(width, result.width());
    try std.testing.expectEqual(height, result.height());
}

test "RtwImage pixel data" {
    const path = try absolutePath(std.testing.allocator, "assets/earthmap.jpg");
    defer std.testing.allocator.free(path);

    var result = try RtwImage.init(std.testing.allocator);
    defer result.destroy(std.testing.allocator);

    try result.load(path);

    const firstPixel = result.pixelData(100, 100);
    _ = firstPixel;

    //try std.testing.expectEqual(Color{ 0, 0, 0 }, firstPixel);
}
