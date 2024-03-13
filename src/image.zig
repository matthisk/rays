const std = @import("std");
const c = @import("c.zig");

pub const RtwImage = struct {
    image_width: c_int = 0,
    image_height: c_int = 0,
    bytes_per_pixel: c_int = 3,
    bytes_per_scanline: c_int = undefined,
    raw: []u8 = undefined,

    pub fn init(filename: []const u8) RtwImage {
        var result = RtwImage{};

        _ = result.load(filename);

        return result;
    }

    pub fn load(self: *RtwImage, filename: []const u8) bool {
        var image_data = c.stbi_load(filename.ptr, &self.image_width, &self.image_height, &self.bytes_per_pixel, @as(c_int, @intCast(self.bytes_per_pixel)));
        self.bytes_per_scanline = self.image_width * self.bytes_per_pixel;

        if (image_data == null) {
            std.debug.print("failed to load image {s}\n", .{c.stbi_failure_reason()});
        }

        return image_data != null;
    }
};

test "RtwImage" {
    var cwd_path = try std.fs.cwd().realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(cwd_path);

    var path = try std.fs.path.join(std.testing.allocator, &[_][]const u8{ cwd_path, "assets/image.png" });
    defer std.testing.allocator.free(path);

    std.debug.print("path {s}\n", .{path});

    _ = RtwImage.init(path);
}
