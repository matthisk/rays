const std = @import("std");
const Color = @import("color.zig").Color;
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});
const assert = @import("std").debug.assert;

const SDL_WINDOWPOS_UNDEFINED = @as(c_int, @bitCast(c.SDL_WINDOWPOS_UNDEFINED_MASK));

pub fn initialize(w: i32, h: i32, allocator: std.mem.Allocator, renderFn: anytype) !void {
    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
        c.SDL_Log("Unable to initialize SDL: %s", c.SDL_GetError());
        return error.SDLInitializationFailed;
    }
    defer c.SDL_Quit();

    const window = c.SDL_CreateWindow("weekend raytracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, w, h, c.SDL_WINDOW_OPENGL) orelse {
        c.SDL_Log("Unable to create window: %s", c.SDL_GetError());
        return error.SDLInitializationFailed;
    };

    const surface = c.SDL_GetWindowSurface(window) orelse {
        c.SDL_Log("Unable to get window surface: %s", c.SDL_GetError());
        return error.SDLInitializationFailed;
    };

    if (c.SDL_UpdateWindowSurface(window) != 0) {
        c.SDL_Log("Error updating window surface: %s", c.SDL_GetError());
        return error.SDLUpdateWindowFailed;
    }

    const thread = try std.Thread.spawn(.{ .allocator = allocator }, renderFn, .{ window, surface });

    renderBlank(w, h, surface);

    if (c.SDL_UpdateWindowSurface(window) != 0) {
        c.SDL_Log("Error updating window surface: %s", c.SDL_GetError());
        return error.SDLUpdateWindowFailed;
    }

    var running = true;
    while (running) {
        var event: c.SDL_Event = undefined;
        while (c.SDL_PollEvent(&event) != 0) {
            switch (event.type) {
                c.SDL_QUIT => {
                    running = false;
                },
                else => {},
            }
        }

        c.SDL_Delay(16);
    }

    // TODO: We quit the program, so we can terminate this thread.
    thread.detach();

    c.SDL_DestroyWindow(window);
    c.SDL_Quit();
}

fn renderBlank(w: i32, h: i32, surface: *c.SDL_Surface) void {
    const color = Color.init(0.5, 1, 1);
    for (0..@intCast(w)) |x| {
        for (0..@intCast(h)) |y| {
            setPixel(surface, @intCast(x), @intCast(y), toBgra(@as(u32, @intFromFloat(255.99 * color.x)), @as(u32, @intFromFloat(255.99 * color.y)), @as(u32, @intFromFloat(255.99 * color.z))));
        }
    }
}

fn setPixel(surf: *c.SDL_Surface, x: c_int, y: c_int, pixel: u32) void {
    const target_pixel = @intFromPtr(surf.pixels) +
        @as(usize, @intCast(y)) * @as(usize, @intCast(surf.pitch)) +
        @as(usize, @intCast(x)) * 4;
    @as(*u32, @ptrFromInt(target_pixel)).* = pixel;
}

fn toBgra(r: u32, g: u32, b: u32) u32 {
    return 255 << 24 | r << 16 | g << 8 | b;
}
