const std = @import("std");
const colors = @import("color.zig");
const c = @import("c.zig");
const assert = @import("std").debug.assert;

const ColorAndSamples = colors.ColorAndSamples;

const SDL_WINDOWPOS_UNDEFINED = @as(c_int, @bitCast(c.SDL_WINDOWPOS_UNDEFINED_MASK));

pub fn initialize(w: i32, h: i32, image_buffer: [][]ColorAndSamples) !void {
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

        renderImageBuffer(w, h, surface, image_buffer);

        if (c.SDL_UpdateWindowSurface(window) != 0) {
            c.SDL_Log("Error updating window surface: %s", c.SDL_GetError());
        }

        c.SDL_Delay(16);
    }

    c.SDL_DestroyWindow(window);
    c.SDL_Quit();

    std.os.exit(0);
}

fn renderImageBuffer(w: i32, h: i32, surface: *c.SDL_Surface, image_buffer: [][]ColorAndSamples) void {
    for (0..@intCast(w)) |x| {
        for (0..@intCast(h)) |y| {
            setPixel(surface, @intCast(x), @intCast(y), colors.toBgra(colors.toGamma(image_buffer[x][y])));
        }
    }
}

fn setPixel(surf: *c.SDL_Surface, x: c_int, y: c_int, pixel: u32) void {
    const target_pixel = @intFromPtr(surf.pixels) +
        @as(usize, @intCast(y)) * @as(usize, @intCast(surf.pitch)) +
        @as(usize, @intCast(x)) * 4;
    @as(*u32, @ptrFromInt(target_pixel)).* = pixel;
}
