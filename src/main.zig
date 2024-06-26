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
const DiffuseLight = @import("material.zig").DiffuseLight;
const Metal = @import("material.zig").Metal;
const Hittable = @import("objects.zig").Hittable;
const HittableList = @import("objects.zig").HittableList;
const HitRecord = @import("objects.zig").HitRecord;
const Sphere = @import("objects.zig").Sphere;
const Planar = @import("objects.zig").Planar;
const Translate = @import("objects.zig").Translate;
const Rotate = @import("objects.zig").Rotate;
const BvhTree = @import("objects.zig").BvhTree;
const Cilinder = @import("objects.zig").Cilinder;
const printPpmToStdout = @import("stdout.zig").printPpmToStdout;
const CheckerTexture = @import("texture.zig").CheckerTexture;
const ImageTexture = @import("texture.zig").ImageTexture;
const NoiseTexture = @import("texture.zig").NoiseTexture;
const MarbleTexture = @import("texture.zig").MarbleTexture;
const WoodTexture = @import("texture.zig").WoodTexture;
const RtwImage = @import("image.zig").RtwImage;
const Perlin = @import("perlin.zig");

const degreesToRadians = @import("math.zig").degreesToRadians;
const linearToGamma = @import("math.zig").linearToGamma;
const absolutePath = @import("util.zig").absolutePath;

const Vector3 = vector.Vector3;
const Ray = rays.Ray;
const Color = colors.Color;
const ColorAndSamples = colors.ColorAndSamples;

const ObjectList = std.ArrayList(Hittable);

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var arena = std.heap.ArenaAllocator.init(gpa.allocator());
var allocator = arena.allocator();

const image_width: u32 = 600;
const image_height: u32 = 600;

const number_of_threads = 32;

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
        .writer = try SharedStateImageWriter.init(allocator, image_buffer),
    };

    // Generate a random world.
    const world = try scene(7, &camera, &objects);

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

    try window.initialize(image_width, image_height, image_buffer);

    for (threads.items) |thread| {
        thread.join();
    }

    // try printPpmToStdout(image_buffer);
}

const Task = struct {
    thread_idx: u32,
    chunk_size: u32,
    world: Hittable,
    camera: *Camera,
};

pub fn renderFn(task: Task) !void {
    try task.camera.render(task);
}

fn scene(i: usize, camera: *Camera, objects: *ObjectList) !Hittable {
    try switch (i) {
        0 => randomSpheresScene(camera, objects),
        1 => twoSpheresScene(camera, objects),
        2 => globeScene(camera, objects),
        3 => perlinScene(camera, objects),
        4 => quadsScene(camera, objects),
        5 => simpleLight(camera, objects),
        6 => cornellBox(camera, objects),
        7 => lego(camera, objects),
        8 => legoScene(camera, objects),
        else => cylinderScene(camera, objects),
    };

    const tree = try BvhTree.init(allocator, objects.items, 0, objects.items.len);
    return Hittable{ .tree = tree };
}

fn perlinScene(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.vfov = 20;
    camera.lookfrom = Vector3{ 13, 2, 3 };
    camera.lookat = Vector3{ 0, 0, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0.70, 0.80, 1.00 };

    const perlin = try Perlin.init(allocator);
    const ground_texture = CheckerTexture.initWithColors(0.32, Color{ 0.2, 0.3, 0.1 }, Color{ 0.9, 0.9, 0.9 });
    const marble_texture = MarbleTexture.init(perlin);
    const wood_texture = WoodTexture.init(perlin);

    try objects.append(Sphere.init(Vector3{ 0, -1001, 0 }, 1000, Lambertian.initWithTexture(ground_texture)));
    try objects.append(Sphere.init(Vector3{ -4, 1, -6 }, 2, Lambertian.initWithTexture(marble_texture)));
    try objects.append(Sphere.init(Vector3{ -6, 0.8, -2 }, 1.8, Lambertian.initWithTexture(wood_texture)));
}

fn quadsScene(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.vfov = 80;
    camera.lookfrom = Vector3{ 0, 0, 9 };
    camera.lookat = Vector3{ 0, 0, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0.70, 0.80, 1.00 };

    const left_red = Lambertian.init(Color{ 1.0, 0.2, 0.2 });
    const back_green = Lambertian.init(Color{ 0.2, 1.0, 0.2 });
    const right_blue = Lambertian.init(Color{ 0.2, 0.2, 1.0 });
    const upper_orange = Lambertian.init(Color{ 1.0, 0.5, 0 });
    const lower_teal = Lambertian.init(Color{ 0.2, 0.8, 0.8 });

    try objects.append(Planar.initQuad(Vector3{ -3, -2, 5 }, Vector3{ 0, 0, -4 }, Vector3{ 0, 4, 0 }, left_red));
    try objects.append(Planar.initTriangle(Vector3{ -2, -2, 0 }, Vector3{ 4, 0, 0 }, Vector3{ 0, 4, 0 }, back_green));
    try objects.append(Planar.initQuad(Vector3{ 3, -2, 1 }, Vector3{ 0, 0, 4 }, Vector3{ 0, 4, 0 }, right_blue));
    try objects.append(Planar.initDisk(Vector3{ 0, 4, 0 }, 2, Vector3{ 0, 1, -0.2 }, upper_orange));
    try objects.append(Planar.initQuad(Vector3{ -2, -3, 5 }, Vector3{ 4, 0, 0 }, Vector3{ 0, 0, -4 }, lower_teal));
}

fn globeScene(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.vfov = 20;
    camera.lookfrom = Vector3{ 0, 0, 12 };
    camera.lookat = Vector3{ 0, 0, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0.70, 0.80, 1.00 };

    const image = try RtwImage.init(allocator);
    const path = try absolutePath(allocator, "assets/earthmap.jpg");
    try image.load(path);

    const imageTexture = ImageTexture.init(image);
    const material = Lambertian.initWithTexture(imageTexture);
    const material_ground = Lambertian.init(Color{ 0.1, 0.1, 0.5 });
    const reflective = Metal.init(Color{ 0.7, 0.6, 0.5 }, 0.01);

    try objects.append(Sphere.init(Vector3{ 0, -1000, -100 }, 1000, material_ground));
    try objects.append(Sphere.init(Vector3{ 0, 0, 1 }, 1.5, material));
    try objects.append(Sphere.init(Vector3{ 3, -1, -5 }, 1, reflective));
}

fn cylinderScene(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.vfov = 20;
    camera.lookfrom = Vector3{ 0, 1, 12 };
    camera.lookat = Vector3{ 0, 0, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0.70, 0.80, 1.00 };
    camera.max_depth = 3;

    const image = try RtwImage.init(allocator);
    const path = try absolutePath(allocator, "assets/earthmap.jpg");
    try image.load(path);
    const imageTexture = ImageTexture.init(image);
    const material = Lambertian.initWithTexture(imageTexture);
    const material_ground = Lambertian.init(Color{ 0.1, 0.1, 0.5 });
    const reflective = Metal.init(Color{ 0.7, 0.6, 0.5 }, 0.01);

    try objects.append(Sphere.init(Vector3{ 0, -1000, -100 }, 1000, material_ground));
    try objects.append(Cilinder.init(Vector3{ -1, -1, -5 }, vector.unitVector(Vector3{ 1, 0.1, 0.3 }), 2, 1, material));
    //try objects.append(Sphere.init(Vector3{ 0, 0, -5 }, 1.5, material));
    try objects.append(Sphere.init(Vector3{ 3, -1, -5 }, 1, reflective));
}

fn twoSpheresScene(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.lookfrom = Vector3{ 13, 2, 3 };
    camera.lookat = Vector3{ 0, 0, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0.70, 0.80, 1.00 };

    const checker = CheckerTexture.initWithColors(0.32, Color{ 0.2, 0.3, 0.1 }, Color{ 0.9, 0.9, 0.9 });
    const material = Lambertian.initWithTexture(checker);

    try objects.append(Sphere.init(Vector3{ 0, -10, 0 }, 10, material));
    try objects.append(Sphere.init(Vector3{ 0, 10, 0 }, 10, material));
}

fn randomSpheresScene(camera: *Camera, objects: *ObjectList) !void {
    camera.background = Color{ 0.70, 0.80, 1.00 };

    const checker = CheckerTexture.initWithColors(0.32, Color{ 0.2, 0.3, 0.1 }, Color{ 0.9, 0.9, 0.9 });
    const material_ground = Lambertian.initWithTexture(checker);

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
                    const center_2 = center + Vector3{ 0, rand.randomBetween(0, radius), 0 };
                    sphere_material = Lambertian.init(colors.randomColorFromPalette());
                    try objects.append(Sphere.initWithMotion(center, center_2, radius, sphere_material));
                } else if (choose_mat < 0.95) {
                    const fuzz = rand.randomBetween(0.0, 0.5);
                    sphere_material = Material{ .metal = Metal{ .albedo = colors.randomColorFromPalette(), .fuzz = fuzz } };
                    try objects.append(Sphere.init(center, radius, sphere_material));
                } else {
                    sphere_material = Material{ .dielectric = Dielectric{ .index_of_refraction = 1.5 } };
                    try objects.append(Sphere.init(center, radius, sphere_material));
                }
            }
        }
    }

    const sphere_material_1 = Material{ .dielectric = Dielectric{ .index_of_refraction = 1.5 } };
    try objects.append(Sphere.init(Vector3{ 0, 1, 0 }, 1, sphere_material_1));
    const sphere_material_2 = Lambertian.init(Vector3{ 0.4, 0.2, 0.1 });
    try objects.append(Sphere.init(Vector3{ -4, 1, 0 }, 1, sphere_material_2));
    const sphere_material_3 = Material{ .metal = Metal{ .albedo = Vector3{ 0.7, 0.6, 0.5 }, .fuzz = 0.0 } };
    try objects.append(Sphere.init(Vector3{ 4, 1, 0 }, 1, sphere_material_3));
}

fn simpleLight(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.vfov = 20;
    camera.lookfrom = Vector3{ 26, 3, 6 };
    camera.lookat = Vector3{ 0, 2, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0, 0, 0 };

    const perlin = try Perlin.init(allocator);
    const marble_texture = MarbleTexture.init(perlin);

    try objects.append(Sphere.init(Vector3{ 0, -1000, 0 }, 1000, Lambertian.initWithTexture(marble_texture)));
    try objects.append(Sphere.init(Vector3{ 0, 2, 0 }, 2, Lambertian.initWithTexture(marble_texture)));

    const light = DiffuseLight.init(Color{ 1, 4, 4 });

    try objects.append(Sphere.init(Vector3{ 0, 7, 2 }, 1, light));
    try objects.append(Planar.initQuad(Vector3{ 3, 1, -2 }, Vector3{ 2, 0, 0 }, Vector3{ 0, 2, 0 }, light));
}

fn cornellBox(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 200;
    camera.max_depth = 50;
    camera.vfov = 40;
    camera.lookfrom = Vector3{ 278, 278, -800 };
    camera.lookat = Vector3{ 278, 278, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0, 0, 0 };

    const red = Lambertian.init(Color{ 0.65, 0.05, 0.05 });
    const white = Lambertian.init(Color{ 0.73, 0.73, 0.73 });
    const green = Lambertian.init(Color{ 0.12, 0.45, 0.15 });
    const light = DiffuseLight.init(Color{ 15, 15, 15 });

    try objects.append(Planar.initQuad(Vector3{ 555, 0, 0 }, Vector3{ 0, 555, 0 }, Vector3{ 0, 0, 555 }, green));
    try objects.append(Planar.initQuad(Vector3{ 0, 0, 0 }, Vector3{ 0, 555, 0 }, Vector3{ 0, 0, 555 }, red));
    try objects.append(Planar.initQuad(Vector3{ 343, 554, 332 }, Vector3{ -160, 0, 0 }, Vector3{ 0, 0, -155 }, light));
    try objects.append(Planar.initQuad(Vector3{ 0, 0, 0 }, Vector3{ 555, 0, 0 }, Vector3{ 0, 0, 555 }, white));
    try objects.append(Planar.initQuad(Vector3{ 555, 555, 555 }, Vector3{ -555, 0, 0 }, Vector3{ 0, 0, -555 }, white));
    try objects.append(Planar.initQuad(Vector3{ 0, 0, 555 }, Vector3{ 555, 0, 0 }, Vector3{ 0, 555, 0 }, white));

    const first_cube = try box(Vector3{ 0, 0, 0 }, Vector3{ 165, 330, 165 }, white);
    const second_cube = try box(Vector3{ 0, 0, 0 }, Vector3{ 165, 165, 165 }, white);

    const first_cube_rotated = try allocator.create(Hittable);
    const second_cube_rotated = try allocator.create(Hittable);
    first_cube_rotated.* = Rotate.init(.y, first_cube, 15);
    second_cube_rotated.* = Rotate.init(.y, second_cube, -18);
    try objects.append(Translate.init(first_cube_rotated, Vector3{ 265, 0, 295 }));
    try objects.append(Translate.init(second_cube_rotated, Vector3{ 130, 0, 65 }));
}

fn lego(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 500;
    camera.max_depth = 50;
    camera.vfov = 40;
    camera.lookfrom = Vector3{ 278, 278, -800 };
    camera.lookat = Vector3{ 278, 278, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0, 0, 0 };

    const red = Lambertian.init(Color{ 0.65, 0.05, 0.05 });
    const green = Lambertian.init(Color{ 0.12, 0.45, 0.15 });
    const yellow = Lambertian.init(Color{ 0.96, 0.8, 0.26 });
    const light = DiffuseLight.init(Color{ 15, 15, 15 });

    var list = ObjectList.init(allocator);

    const palette = [3]Material{ red, green, yellow };

    for (0..20) |i| {
        const ii = @as(f64, @floatFromInt(i));
        for (0..20) |j| {
            const jj = @as(f64, @floatFromInt(j));
            const y: f64 = if (rand.randomFloat() < 0.5) 0 else 96;
            const mat = palette[rand.randomIntBetween(0, 3)];

            const block = try squareBlock(Vector3{ 158 * ii, y, 158 * jj }, 1, mat);

            try list.append(block.*);
        }
    }

    const bricks = try allocator.create(Hittable);
    bricks.* = Hittable{ .tree = try BvhTree.init(allocator, list.items, 0, list.items.len) };

    const rotated_y = try allocator.create(Hittable);
    const rotated_x = try allocator.create(Hittable);
    const translated = try allocator.create(Hittable);

    rotated_y.* = Rotate.init(.y, bricks, -45);
    rotated_x.* = Rotate.init(.x, rotated_y, 10);
    translated.* = Translate.init(rotated_x, Vector3{ -100, -500, 0 });

    try objects.append(translated.*);
    try objects.append(Planar.initQuad(Vector3{ 500, 554, 554 }, Vector3{ -450, 0, 0 }, Vector3{ 0, 0, -540 }, light));

    const glass = Material{ .dielectric = Dielectric{ .index_of_refraction = 1.5 } };
    try objects.append(Sphere.init(Vector3{ 178, 155, 500 }, 150, glass));
}

fn legoScene(camera: *Camera, objects: *ObjectList) !void {
    camera.samples_per_pixel = 200;
    camera.max_depth = 50;
    camera.vfov = 40;
    camera.lookfrom = Vector3{ 478, 278, -600 };
    camera.lookat = Vector3{ 278, 278, 0 };
    camera.vup = Vector3{ 0, 1, 0 };
    camera.defocus_angle = 0;
    camera.background = Color{ 0, 0, 0 };

    const ground = Lambertian.init(Color{ 0.48, 0.83, 0.53 });
    const boxes_per_side = 20;

    for (0..boxes_per_side) |i| {
        const ii = @as(f32, @floatFromInt(i));
        for (0..boxes_per_side) |j| {
            const jj = @as(f32, @floatFromInt(j));

            const w = 100.0;
            const x0 = -1000.0 + ii * w;
            const z0 = -1000.0 + jj * w;
            const y0 = 0.0;
            const x1 = x0 + w;
            const y1 = @as(f32, @floatCast(rand.randomBetween(1, 101)));

            const obj = try box(Vector3{ x0, y0, z0 }, Vector3{ x1, y1, z0 + w }, ground);
            try objects.append(obj.*);
        }
    }

    const light = DiffuseLight.init(Color{ 7, 7, 7 });

    try objects.append(Planar.initQuad(Vector3{ 123, 554, 147 }, Vector3{ 300, 0, 0 }, Vector3{ 0, 0, 265 }, light));
}

fn squareBlock(origin: Vector3, scale: f32, mat: Material) !*Hittable {
    const width = 158 * scale;
    const height = 96 * scale;
    const depth = 158 * scale;

    var objects = ObjectList.init(allocator);
    const first_cube = try box(origin, origin + Vector3{ width, height, depth }, mat);

    const normal = Vector3{ 0, 1, 0 };
    const radius = depth * 0.15;
    const bob_height = 0.1875 * height;
    const z_first_bob = depth / 3 - depth / 15;
    const z_second_bob = 2 * depth / 3 + depth / 15;
    const x_offset = width * 0.5;
    const x_start = (width - x_offset) / 2;

    try objects.append(Cilinder.init(origin + Vector3{ x_start, height, z_first_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start, height, z_second_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start + x_offset, height, z_first_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start + x_offset, height, z_second_bob }, normal, bob_height, radius, mat));
    try objects.append(first_cube.*);

    const result = try allocator.create(Hittable);
    result.* = HittableList.init(objects.items);
    return result;
}

fn rectangleBlock(origin: Vector3, scale: f32, mat: Material) !*Hittable {
    const width = 318 * scale;
    const height = 96 * scale;
    const depth = 158 * scale;

    var objects = ObjectList.init(allocator);
    const first_cube = try box(origin, origin + Vector3{ width, height, depth }, mat);

    const normal = Vector3{ 0, 1, 0 };
    const radius = depth * 0.15;
    const bob_height = 0.1875 * height;
    const z_first_bob = depth / 3 - depth / 15;
    const z_second_bob = 2 * depth / 3 + depth / 15;
    const x_offset = width / 3.975;
    const x_start = (width - 3 * x_offset) / 2;

    try objects.append(Cilinder.init(origin + Vector3{ x_start, height, z_first_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start, height, z_second_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start + x_offset, height, z_first_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start + x_offset, height, z_second_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start + 2 * x_offset, height, z_first_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start + 2 * x_offset, height, z_second_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start + 3 * x_offset, height, z_first_bob }, normal, bob_height, radius, mat));
    try objects.append(Cilinder.init(origin + Vector3{ x_start + 3 * x_offset, height, z_second_bob }, normal, bob_height, radius, mat));
    try objects.append(first_cube.*);

    const result = try allocator.create(Hittable);
    result.* = HittableList.init(objects.items);
    return result;
}

fn box(a: Vector3, b: Vector3, mat: Material) !*Hittable {
    var sides = ObjectList.init(allocator);

    const min = Vector3{ @min(a[0], b[0]), @min(a[1], b[1]), @min(a[2], b[2]) };
    const max = Vector3{ @max(a[0], b[0]), @max(a[1], b[1]), @max(a[2], b[2]) };

    const dx = Vector3{ max[0] - min[0], 0, 0 };
    const dy = Vector3{ 0, max[1] - min[1], 0 };
    const dz = Vector3{ 0, 0, max[2] - min[2] };

    try sides.append(Planar.initQuad(Vector3{ min[0], min[1], max[2] }, dx, dy, mat)); // front
    try sides.append(Planar.initQuad(Vector3{ max[0], min[1], max[2] }, -dz, dy, mat)); // right
    try sides.append(Planar.initQuad(Vector3{ max[0], min[1], min[2] }, -dx, dy, mat)); // back
    try sides.append(Planar.initQuad(min, dz, dy, mat)); // left
    try sides.append(Planar.initQuad(Vector3{ min[0], max[1], max[2] }, dx, -dz, mat)); // top
    try sides.append(Planar.initQuad(min, dx, dz, mat)); // bottom

    const result = try allocator.create(Hittable);
    result.* = HittableList.init(sides.items);

    return result;
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

    background: ?Color = Color{ 0, 0, 0 }, // Scene background color

    u: Vector3 = undefined,
    v: Vector3 = undefined,
    w: Vector3 = undefined,

    writer: *ImageWriter = undefined,

    pub fn render(self: *Camera, task: Task) std.fs.File.Writer.Error!void {
        try self.initialize();

        const start_at = task.thread_idx * task.chunk_size;
        const end_before = start_at + task.chunk_size;

        for (1..self.samples_per_pixel + 1) |number_of_samples| {
            for (start_at..end_before) |i| {
                const x = @mod(i, self.img_width) + 1;
                const y = @divTrunc(i, self.img_width) + 1;

                const ray = self.getRay(x, y);
                const color = self.rayColor(ray, task.world, self.max_depth);

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
            var color_from_emmission = hit_record.mat.emitted(hit_record.u, hit_record.v, hit_record.p);

            if (hit_record.mat.scatter(ray, hit_record, &attenuation, &scattered)) {
                return color_from_emmission + attenuation * self.rayColor(scattered, world, depth - 1);
            }

            return color_from_emmission;
        }

        if (self.background) |background| {
            return background;
        } else {
            // Sky model.
            // This section calculates the color of a ray in case it doesn't hit any object.
            const unit_direction = vector.unitVector(ray.direction);
            const a = 0.5 * (unit_direction[1] + 1.0);

            return Color{ 1, 1, 1 } * vector.splat3(1.0 - a) + Color{ 0.5, 0.7, 1.0 } * vector.splat3(a);
        }
    }

    fn getRay(self: *Camera, x: u64, y: u64) Ray {
        // Get a randomly-sampled camera ray for the pixel at location x,y, originating from
        // the camera defocus disk.
        const pixel_center = self.pixel00_loc + self.pixel_delta_u * vector.splat3(@floatFromInt(x)) + self.pixel_delta_v * vector.splat3(@floatFromInt(y));
        const pixel_sample = pixel_center + self.pixelSampleSquare();

        const ray_origin = if (self.defocus_angle <= 0) self.center else self.defocusDiskSample();
        const ray_direction = pixel_sample - ray_origin;
        const time = rand.randomFloat();

        return Ray.initWithTime(ray_origin, ray_direction, time);
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
    shared_state: *SharedStateImageWriter,
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

const AtomicU64 = std.atomic.Atomic(u64);

const SharedStateImageWriter = struct {
    buffer: [][]ColorAndSamples,
    samples: AtomicU64,
    total_samples: u64,

    pub fn init(alloc: std.mem.Allocator, buffer: [][]ColorAndSamples) !*ImageWriter {
        const result = try alloc.create(ImageWriter);
        const writer = try alloc.create(SharedStateImageWriter);
        writer.* = SharedStateImageWriter{ .buffer = buffer, .samples = AtomicU64.init(0), .total_samples = image_width * image_height * 500 };
        result.* = ImageWriter{ .shared_state = writer };
        return result;
    }

    pub fn writeColor(self: *SharedStateImageWriter, x: u64, y: u64, color: Color, number_of_samples: u64) !void {
        _ = self.samples.fetchAdd(1, std.atomic.Ordering.Monotonic);
        self.buffer[x][y] += vector.Vector4{ color[0], color[1], color[2], 0 };
        self.buffer[x][y][3] = @floatFromInt(number_of_samples);

        const samples = self.samples.load(std.atomic.Ordering.Monotonic);
        if (samples % 1_000_000 == 0) {
            std.debug.print("{d:.0}\n", .{100 * @as(f32, @floatFromInt(samples)) / @as(f32, @floatFromInt(self.total_samples))});
        }
    }
};

test {
    std.testing.refAllDeclsRecursive(@This());
}
