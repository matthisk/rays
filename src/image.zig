const std = @import("std");
const c = @import("c.zig");
const absolutePath = @import("util.zig").absolutePath;

pub const RtwImage = struct {
    image_width: c_int = 0,
    image_height: c_int = 0,
    bytes_per_pixel: c_int = 3,
    bytes_per_scanline: c_int = undefined,
    raw: []u8 = undefined,

    pub fn init(allocator: std.mem.Allocator, filename: [:0]const u8) !*RtwImage {
        var result = try allocator.create(RtwImage);

        _ = result.load(filename);

        return result;
    }

    fn load(self: *RtwImage, filename: [:0]const u8) bool {
        var image_data = c.stbi_load(filename.ptr, &self.image_width, &self.image_height, &self.bytes_per_pixel, self.bytes_per_pixel);
        self.bytes_per_scanline = self.image_width * self.bytes_per_pixel;

        // Convert c pointer to a zig slice.
        self.raw = image_data[0..@as(usize, @intCast(self.bytes_per_scanline * self.image_height))];

        if (image_data == null) {
            std.debug.print("failed to load image {s}\n", .{c.stbi_failure_reason()});
        }

        return image_data != null;
    }
};

test "RtwImage" {
    const path = try absolutePath(std.testing.allocator, "assets/image.png");
    defer std.testing.allocator.free(path);

    var result = try RtwImage.init(std.testing.allocator, path);
    std.testing.allocator.destroy(result);
}
