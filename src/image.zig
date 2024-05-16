const std = @import("std");
const c = @import("c.zig");
const absolutePath = @import("util.zig").absolutePath;

const ImageReadError = error{FailedToReadImage};

pub const RtwImage = struct {
    image_width: c_int = 0,
    image_height: c_int = 0,
    bytes_per_pixel: c_int = 0,
    bytes_per_scanline: c_int = undefined,
    raw: [*c]u8 = undefined,
    image_data: []u8 = undefined,

    pub fn init(allocator: std.mem.Allocator) !*RtwImage {
        var result = try allocator.create(RtwImage);
        result.* = RtwImage{};

        return result;
    }

    pub fn load(self: *RtwImage, filename: [:0]const u8) ImageReadError!void {
        self.raw = c.stbi_load(filename.ptr, &self.image_width, &self.image_height, &self.bytes_per_pixel, self.bytes_per_pixel);

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
};

test "RtwImage" {
    const path = try absolutePath(std.testing.allocator, "assets/image.png");
    defer std.testing.allocator.free(path);

    var result = try RtwImage.init(std.testing.allocator);
    defer result.destroy(std.testing.allocator);

    try result.load(path);
}
