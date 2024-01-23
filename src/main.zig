const std = @import("std");
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});
const vector = @import("vector.zig");
const rays = @import("ray.zig");
const colors = @import("color.zig");
const Aabb = @import("aabb.zig");
const rand = @import("rand.zig");
const window = @import("window.zig");
const Interval = @import("interval.zig");
const Material = @import("material.zig").Material;
const Dielectric = @import("material.zig").Dielectric;
const Lambertian = @import("material.zig").Lambertian;
const Metal = @import("material.zig").Metal;
const Hittable = @import("objects.zig").Hittable;
const HittableList = @import("objects.zig").HittableList;
const HitRecord = @import("objects.zig").HitRecord;
const Sphere = @import("objects.zig").Sphere;
const BvhTree = @import("objects.zig").BvhTree;
const printPpmToStdout = @import("stdout.zig").printPpmToStdout;

const degreesToRadians = @import("math.zig").degreesToRadians;
const linearToGamma = @import("math.zig").linearToGamma;

const Vector3 = vector.Vector3;
const Ray = rays.Ray;
const Color = colors.Color;
const ColorAndSamples = colors.ColorAndSamples;

const ObjectList = std.ArrayList(Hittable);

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var arena = std.heap.ArenaAllocator.init(gpa.allocator());
var allocator = arena.allocator();

const image_width: u32 = 1920;
const image_height: u32 = 1080;
const aspect_ratio = 16.0 / 9.0;

const number_of_threads = 8;

pub fn main() !void {
    defer arena.deinit();
    const image_buffer = try allocator.alloc([]ColorAndSamples, image_width);

    for (0..image_width) |x| {
        image_buffer[x] = try allocator.alloc(ColorAndSamples, image_height);
    }

    for (0..image_width) |x| {
        for (0..image_height) |y| {
            image_buffer[x][y] = vector.Vector4{ 0, 0, 0, 1 };
        }
    }

    // Allocate heap.
    var objects = ObjectList.init(allocator);
    defer objects.deinit();

    // Generate a random world.
    const world = try generateWorld(&objects);

    // Initialize camera and render frame.
    var camera = Camera{
        // Output.
        .img_width = image_width,
        .img_height = image_height,

        // Render config.
        .samples_per_pixel = 500,
        .max_depth = 16,

        // View.
        .vfov = 20,
        .lookfrom = Vector3{ 13, 2, 3 },
        .lookat = Vector3{ 0, 0, 0 },
        .vup = Vector3{ 0, 1, 0 },

        // Focus.
        .defocus_angle = 0.6,
        .focus_dist = 10.0,

        // Writer.
        .writer = SharedStateImageWriter.init(image_buffer),
    };

    var threads = std.ArrayList(std.Thread).init(allocator);

    for (0..number_of_threads) |thread_idx| {
        const task = Task{
            .thread_idx = @intCast(thread_idx),
            .chunk_size = (image_width * image_height) / number_of_threads,
            .world = world,
            .camera = &camera,
        };

        const thread = try std.Thread.spawn(.{ .allocator = allocator }, renderFn, .{task});

        try threads.append(thread);
    }

    // try window.initialize(image_width, image_height, image_buffer);

    var timer = try std.time.Timer.start();

    for (threads.items) |thread| {
        thread.join();
    }

    try printPpmToStdout(image_buffer);

    std.debug.print("finished render in {d}ms", .{timer.read() / 1000_000});
}

const Task = struct {
    thread_idx: u32,
    chunk_size: u32,
    world: Hittable,
    camera: *Camera,
};

pub fn renderFn(context: Task) !void {
    try context.camera.render(context);
}

fn generateWorld(objects: *ObjectList) !Hittable {
    const material_ground = Material{ .lambertian = Lambertian{ .albedo = Color{ 0.5, 0.5, 0.5 } } };

    try objects.append(Sphere.init(Vector3{ 0, -1000, 0 }, 1000, material_ground));

    for (0..22) |k| {
        for (0..22) |l| {
            const a: f64 = @as(f64, @floatFromInt(k)) - 11;
            const b: f64 = @as(f64, @floatFromInt(l)) - 11;

            const choose_mat = rand.randomFloat();
            const radius = rand.randomBetween(0.1, 0.3);
            const center = Vector3{ a + 0.9 * rand.randomFloat(), radius, b + 0.9 * rand.randomFloat() };

            if (vector.length(center - Vector3{ 4, radius, 0 }) > 0.9) {
                var sphere_material: Material = undefined;

                if (choose_mat < 0.8) {
                    sphere_material = Material{ .lambertian = Lambertian{ .albedo = colors.randomColorFromPalette() } };
                } else if (choose_mat < 0.95) {
                    const fuzz = rand.randomBetween(0.0, 0.5);
                    sphere_material = Material{ .metal = Metal{ .albedo = colors.randomColorFromPalette(), .fuzz = fuzz } };
                } else {
                    sphere_material = Material{ .dielectric = Dielectric{ .index_of_refraction = 1.5 } };
                }

                try objects.append(Sphere.init(center, radius, sphere_material));
            }
        }
    }

    const sphere_material_1 = Material{ .dielectric = Dielectric{ .index_of_refraction = 1.5 } };
    try objects.append(Sphere.init(Vector3{ 0, 1, 0 }, 1, sphere_material_1));
    const sphere_material_2 = Material{ .lambertian = Lambertian{ .albedo = Vector3{ 0.4, 0.2, 0.1 } } };
    try objects.append(Sphere.init(Vector3{ -4, 1, 0 }, 1, sphere_material_2));
    const sphere_material_3 = Material{ .metal = Metal{ .albedo = Vector3{ 0.7, 0.6, 0.5 }, .fuzz = 0.0 } };
    try objects.append(Sphere.init(Vector3{ 4, 1, 0 }, 1, sphere_material_3));

    const tree = try BvhTree.init(allocator, objects.items, 0, objects.items.len);

    return Hittable{ .tree = tree };
}

const Camera = struct {
    aspect_ratio: f64 = 1.0, // Ratio of image width over height
    img_width: u32 = 0, // Rendered image width in pixel count
    img_height: u32 = 0,
    samples_per_pixel: u32 = 100, // Count of random samples for each pixel
    max_depth: u32 = 100, // Maximum number of ray bounces into scene
    vfov: f64 = 90, // Vertical view angle (field of view).
    center: Vector3 = Vector3{ 0, 0, 0 },
    pixel00_loc: Vector3 = Vector3{ 0, 0, 0 },
    pixel_delta_u: Vector3 = Vector3{ 0, 0, 0 },
    pixel_delta_v: Vector3 = Vector3{ 0, 0, 0 },

    lookfrom: Vector3 = Vector3{ 0, 0, -1 }, // Point camera is looking from
    lookat: Vector3 = Vector3{ 0, 0, 0 }, // Point camera is looking at
    vup: Vector3 = Vector3{ 0, 1, 0 }, // Camera-relative "up" direction

    defocus_angle: f64 = 0, // Variation angle of rays through each pixel
    focus_dist: f64 = 10.0, // Distance from camera lookfrom point to plane of perfect focus
    defocus_disk_u: Vector3 = undefined,
    defocus_disk_v: Vector3 = undefined,

    u: Vector3 = undefined,
    v: Vector3 = undefined,
    w: Vector3 = undefined,

    writer: ImageWriter = undefined,

    pub fn render(self: *Camera, context: Task) std.fs.File.Writer.Error!void {
        try self.initialize();

        const start_at = context.thread_idx * context.chunk_size;
        const end_before = start_at + context.chunk_size;

        for (1..self.samples_per_pixel + 1) |number_of_samples| {
            for (start_at..end_before) |i| {
                const x = @mod(i, self.img_width) + 1;
                const y = @divTrunc(i, self.img_width) + 1;

                const ray = self.getRay(x, y);
                const color = self.rayColor(ray, context.world, self.max_depth);

                try self.writer.writeColor(x - 1, y - 1, color, number_of_samples);
            }
        }
    }

    fn initialize(self: *Camera) !void {
        self.center = self.lookfrom;

        // Camera
        const theta = degreesToRadians(self.vfov);
        const h = std.math.tan(theta / 2);
        const viewport_height = 2 * h * self.focus_dist;
        const viewport_aspect_ratio = @as(f64, @floatFromInt(self.img_width)) / @as(f64, @floatFromInt(self.img_height));
        const viewport_width = viewport_height * viewport_aspect_ratio;

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        self.w = vector.unitVector(self.lookfrom - self.lookat);
        self.u = vector.unitVector(vector.cross(self.vup, self.w));
        self.v = vector.cross(self.w, self.u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        const viewport_u = self.u * vector.splat3(viewport_width);
        const viewport_v = self.v * vector.splat3(-viewport_height);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.pixel_delta_u = viewport_u / vector.splat3(@floatFromInt(self.img_width));
        self.pixel_delta_v = viewport_v / vector.splat3(@floatFromInt(self.img_height));

        // Calculate the location of the upper left pixel.
        const viewport_upper_left = self.center - (self.w * vector.splat3(self.focus_dist)) - (viewport_u / vector.splat3(2)) - (viewport_v / vector.splat3(2));
        self.pixel00_loc = viewport_upper_left + self.pixel_delta_u + self.pixel_delta_v * vector.splat3(0.5);

        // Calculate the camera defocus disk basis vectors.
        const defocus_radius = self.focus_dist * std.math.tan(degreesToRadians(self.defocus_angle / 2));
        self.defocus_disk_u = self.u * vector.splat3(defocus_radius);
        self.defocus_disk_v = self.v * vector.splat3(defocus_radius);
    }

    fn rayColor(self: *Camera, ray: Ray, world: Hittable, depth: u32) Color {
        if (depth <= 0)
            return Color{ 0, 0, 0 };

        const opt_hit_record = world.hit(ray, Interval{ .min = 0.001, .max = std.math.inf(f64) });

        if (opt_hit_record) |hit_record| {
            var attenuation: Color = undefined;
            var scattered: Ray = undefined;

            if (hit_record.mat.scatter(ray, hit_record, &attenuation, &scattered)) {
                return attenuation * self.rayColor(scattered, world, depth - 1);
            }

            return Color{ 0, 0, 0 };
        }

        // Sky model.
        // This section calculates the color of a ray in case it doesn't hit any object.
        const unit_direction = vector.unitVector(ray.direction);
        const a = 0.5 * (unit_direction[1] + 1.0);

        return Color{ 1, 1, 1 } * vector.splat3(1.0 - a) + Color{ 0.5, 0.7, 1.0 } * vector.splat3(a);
    }

    fn getRay(self: *Camera, x: u64, y: u64) Ray {
        // Get a randomly-sampled camera ray for the pixel at location x,y, originating from
        // the camera defocus disk.
        const pixel_center = self.pixel00_loc + self.pixel_delta_u * vector.splat3(@floatFromInt(x)) + self.pixel_delta_v * vector.splat3(@floatFromInt(y));
        const pixel_sample = pixel_center + self.pixelSampleSquare();

        const ray_origin = if (self.defocus_angle <= 0) self.center else self.defocusDiskSample();
        const ray_direction = pixel_sample - ray_origin;

        return Ray.init(ray_origin, ray_direction);
    }

    fn pixelSampleSquare(self: *Camera) Vector3 {
        const px = -0.5 * rand.randomFloat();
        const py = -0.5 * rand.randomFloat();

        return self.pixel_delta_u * vector.splat3(px) + self.pixel_delta_v * vector.splat3(py);
    }

    fn defocusDiskSample(self: *Camera) Vector3 {
        const p = vector.randomInUnitDisk();
        return self.center + self.defocus_disk_u * vector.splat3(p[0]) + self.defocus_disk_v * vector.splat3(p[1]);
    }
};

const ImageWriter = union(enum) {
    shared_state: SharedStateImageWriter,
    sink: SinkImageWriter,

    pub fn writeColor(self: ImageWriter, x: u64, y: u64, color: Color, number_of_samples: u64) !void {
        switch (self) {
            inline else => |case| try case.writeColor(x, y, color, number_of_samples),
        }
    }
};

const SinkImageWriter = struct {
    pub fn init() ImageWriter {
        return ImageWriter{ .sink = .{} };
    }

    pub fn writeColor(self: SinkImageWriter, x: u64, y: u64, color: Color, number_of_samples: u64) !void {
        // ignore.
        _ = self;
        _ = x;
        _ = y;
        _ = color;
        _ = number_of_samples;
    }
};

const SharedStateImageWriter = struct {
    buffer: [][]ColorAndSamples,

    pub fn init(buffer: [][]ColorAndSamples) ImageWriter {
        return ImageWriter{ .shared_state = .{
            .buffer = buffer,
        } };
    }

    pub fn writeColor(self: SharedStateImageWriter, x: u64, y: u64, color: Color, number_of_samples: u64) !void {
        self.buffer[x][y] += vector.Vector4{ color[0], color[1], color[2], 0 };
        self.buffer[x][y][3] = @floatFromInt(number_of_samples);
    }
};
