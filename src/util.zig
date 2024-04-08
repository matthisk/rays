const std = @import("std");

pub fn absolutePath(allocator: std.mem.Allocator, input_path: [:0]const u8) ![:0]const u8 {
    var cwd_path = try std.fs.cwd().realpathAlloc(allocator, ".");
    defer allocator.free(cwd_path);

    var path = try std.fs.path.join(allocator, &[_][]const u8{ cwd_path, input_path });
    defer allocator.free(path);

    return allocator.dupeZ(u8, path);
}
