const std = @import("std");
const colors = @import("color.zig");
const ColorAndSamples = colors.ColorAndSamples;

// Print the image buffer to PPM format to the stdout.
pub fn printPpmToStdout(buffer: [][]ColorAndSamples) !void {
    const img_width = buffer.len;
    const img_height = buffer[0].len;

    var stdout = std.io.getStdOut().writer();
    try stdout.print("P3\n{d} {d}\n255\n", .{ img_width, img_height });

    for (0..img_height) |y| {
        for (0..img_width) |x| {
            const color = colors.toGamma(buffer[x][y].color, 1.0 / @as(f32, @floatFromInt(buffer[x][y].number_of_samples)));
            try stdout.print("{d} {d} {d}\t", .{
                @floor(color.x * 255.999),
                @floor(color.y * 255.999),
                @floor(color.z * 255.999),
            });
        }
    }
}
