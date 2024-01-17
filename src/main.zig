const std = @import("std");
const c = @cImport({
    @cInclude("SDL2/SDL.h");
});
const vecs = @import("vec3.zig");
const rays = @import("ray.zig");
const colors = @import("color.zig");
const rand = @import("rand.zig");
const window = @import("window.zig");
const Interval = @import("interval.zig");
const Material = @import("material.zig").Material;
const Dielectric = @import("material.zig").Dielectric;
const Lambertian = @import("material.zig").Lambertian;
const Metal = @import("material.zig").Metal;
const printPpmToStdout = @import("stdout.zig").printPpmToStdout;

const degreesToRadians = @import("math.zig").degreesToRadians;
const linearToGamma = @import("math.zig").linearToGamma;

const Vec3 = vecs.Vec3;
const Ray = rays.Ray;
const Color = colors.Color;
const ColorAndSamples = colors.ColorAndSamples;
const Vector3 = @Vector(3, f64);

const ObjectList = std.ArrayList(Hittable);

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var arena = std.heap.ArenaAllocator.init(gpa.allocator());
var allocator = arena.allocator();

const image_width: u32 = 400;
const image_height: u32 = 225;
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
            image_buffer[x][y] = .{ .color = Color.init(0, 0, 0), .number_of_samples = 1 };
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
        .samples_per_pixel = 1,
        .max_depth = 16,

        // View.
        .vfov = 20,
        .lookfrom = Vec3.init(13, 2, 3),
        .lookat = Vec3.init(0, 0, 0),
        .vup = Vec3.init(0, 1, 0),

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

    for (threads.items) |thread| {
        thread.join();
    }

    try printPpmToStdout(image_buffer);
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
    const material_ground = Material{ .lambertian = Lambertian{ .albedo = Color.init(0.5, 0.5, 0.5) } };

    try objects.append(Hittable{ .sphere = Sphere{ .center = Vec3.init(0, -1000, 0), .radius = 1000, .mat = material_ground } });

    for (0..22) |k| {
        for (0..22) |l| {
            const a: f64 = @as(f64, @floatFromInt(k)) - 11;
            const b: f64 = @as(f64, @floatFromInt(l)) - 11;

            const choose_mat = rand.randomFloat();
            const center = Vec3.init(a + 0.9 * rand.randomFloat(), 0.2, b + 0.9 * rand.randomFloat());

            if (center.minus(Vec3.init(4, 0.2, 0)).length() > 0.9) {
                var sphere_material: Material = undefined;

                if (choose_mat < 0.8) {
                    sphere_material = Material{ .lambertian = Lambertian{ .albedo = vecs.random().multiplyByVec3(vecs.random()) } };
                } else if (choose_mat < 0.95) {
                    const fuzz = rand.randomBetween(0.0, 0.5);
                    sphere_material = Material{ .metal = Metal{ .albedo = vecs.random().multiplyByVec3(vecs.randomBetween(0.5, 1)), .fuzz = fuzz } };
                } else {
                    sphere_material = Material{ .dielectric = Dielectric{ .index_of_refraction = 1.5 } };
                }

                try objects.append(Hittable{ .sphere = Sphere{ .center = center, .radius = 0.2, .mat = sphere_material } });
            }
        }
    }

    const sphere_material = Material{ .dielectric = Dielectric{ .index_of_refraction = 1.5 } };
    try objects.append(Hittable{ .sphere = Sphere{ .center = Vec3.init(0, 1.5, 0), .radius = 1.5, .mat = sphere_material } });
    try objects.append(Hittable{ .sphere = Sphere{ .center = Vec3.init(0, 1.5, 0), .radius = -1.3, .mat = sphere_material } });

    const sphere_material_2 = Material{ .lambertian = Lambertian{ .albedo = vecs.random().multiplyByVec3(vecs.random()) } };
    try objects.append(Hittable{ .sphere = Sphere{ .center = Vec3.init(1, 1.5, 1.5), .radius = 1.5, .mat = sphere_material_2 } });

    return Hittable{ .list = HittableList{ .objects = objects.items } };
}

pub const HitRecord = struct {
    p: Vec3 = Vec3.init(0, 0, 0),
    normal: Vec3 = Vec3.init(0, 0, 0),
    t: f64 = 0,
    front_face: bool = false,
    mat: Material = undefined,

    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.
    pub fn setFaceNormal(self: *HitRecord, ray: Ray, outward_normal: Vec3) void {
        self.front_face = ray.direction.dot(outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else outward_normal.multiply(-1);
    }
};

const Hittable = union(enum) {
    sphere: Sphere,
    list: HittableList,

    pub fn hit(self: Hittable, ray: Ray, ray_t: Interval) ?HitRecord {
        return switch (self) {
            inline else => |case| case.hit(ray, ray_t),
        };
    }
};

const Sphere = struct {
    center: Vec3,
    radius: f64,
    mat: Material,

    pub fn hit(self: Sphere, ray: Ray, ray_t: Interval) ?HitRecord {
        const oc = ray.origin.minus(self.center);
        const a = ray.direction.lengthSquared();
        const half_b = oc.dot(ray.direction);
        const cc = oc.lengthSquared() - self.radius * self.radius;
        const discriminant = half_b * half_b - a * cc;
        if (discriminant < 0) return null;
        const sqrtd = std.math.sqrt(discriminant);

        var root = (-half_b - sqrtd) / a;

        if (!ray_t.surrounds(root)) {
            root = (-half_b + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return null;
            }
        }

        var hit_record = HitRecord{};
        hit_record.t = root;
        hit_record.p = ray.at(hit_record.t);
        hit_record.mat = self.mat;
        const outward_normal = hit_record.p.minus(self.center).divide(self.radius);
        hit_record.setFaceNormal(ray, outward_normal);

        return hit_record;
    }
};

const HittableList = struct {
    objects: []const Hittable,

    pub fn hit(self: HittableList, ray: Ray, ray_t: Interval) ?HitRecord {
        var latest: ?HitRecord = null;
        var closest_so_far = ray_t.max;

        for (self.objects) |object| {
            const hit_record = object.hit(ray, Interval{ .min = ray_t.min, .max = closest_so_far });
            if (hit_record) |hr| {
                closest_so_far = hr.t;
                latest = hr;
            }
        }

        return latest;
    }
};

const Camera = struct {
    aspect_ratio: f64 = 1.0, // Ratio of image width over height
    img_width: u32 = 0, // Rendered image width in pixel count
    img_height: u32 = 0,
    samples_per_pixel: u32 = 100, // Count of random samples for each pixel
    max_depth: u32 = 100, // Maximum number of ray bounces into scene
    vfov: f64 = 90, // Vertical view angle (field of view).
    center: Vec3 = Vec3.init(0, 0, 0),
    pixel00_loc: Vec3 = Vec3.init(0, 0, 0),
    pixel_delta_u: Vec3 = Vec3.init(0, 0, 0),
    pixel_delta_v: Vec3 = Vec3.init(0, 0, 0),

    lookfrom: Vec3 = Vec3.init(0, 0, -1), // Point camera is looking from
    lookat: Vec3 = Vec3.init(0, 0, 0), // Point camera is looking at
    vup: Vec3 = Vec3.init(0, 1, 0), // Camera-relative "up" direction

    defocus_angle: f64 = 0, // Variation angle of rays through each pixel
    focus_dist: f64 = 10.0, // Distance from camera lookfrom point to plane of perfect focus
    defocus_disk_u: Vec3 = undefined,
    defocus_disk_v: Vec3 = undefined,

    u: Vec3 = undefined,
    v: Vec3 = undefined,
    w: Vec3 = undefined,

    writer: SharedStateImageWriter = undefined,

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
        self.w = vecs.unitVector(self.lookfrom.minus(self.lookat));
        self.u = vecs.unitVector(self.vup.cross(self.w));
        self.v = self.w.cross(self.u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        const viewport_u = self.u.multiply(viewport_width);
        const viewport_v = self.v.multiply(-viewport_height);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.pixel_delta_u = viewport_u.divide(@floatFromInt(self.img_width));
        self.pixel_delta_v = viewport_v.divide(@floatFromInt(self.img_height));

        // Calculate the location of the upper left pixel.
        const viewport_upper_left = self.center.minus(self.w.multiply(self.focus_dist)).minus(viewport_u.divide(2)).minus(viewport_v.divide(2));
        self.pixel00_loc = viewport_upper_left.plus(self.pixel_delta_u.plus(self.pixel_delta_v).multiply(0.5));

        // Calculate the camera defocus disk basis vectors.
        const defocus_radius = self.focus_dist * std.math.tan(degreesToRadians(self.defocus_angle / 2));
        self.defocus_disk_u = self.u.multiply(defocus_radius);
        self.defocus_disk_v = self.v.multiply(defocus_radius);
    }

    fn rayColor(self: *Camera, ray: Ray, world: Hittable, depth: u32) Color {
        if (depth <= 0)
            return Color.init(0, 0, 0);

        const opt_hit_record = world.hit(ray, Interval{ .min = 0.001, .max = std.math.inf(f64) });

        if (opt_hit_record) |hit_record| {
            var attenuation: Color = undefined;
            var scattered: Ray = undefined;

            if (hit_record.mat.scatter(ray, hit_record, &attenuation, &scattered)) {
                return attenuation.multiplyByVec3(self.rayColor(scattered, world, depth - 1));
            }

            return Color.init(0, 0, 0);
        }

        const unit_direction = vecs.unitVector(ray.direction);
        const a = 0.5 * (unit_direction.y + 1.0);

        // const vec = Vector3{ 1, 1, 1 } * @as(Vector3, @splat(1.0 - a)) + Vector3{ 0.5, 0.7, 1.0 } * @as(Vector3, @splat(a));
        // return Color.init(vec[0], vec[1], vec[2]);
        return Color.init(1.0, 1.0, 1.0).multiply(1.0 - a).plus(Color.init(0.5, 0.7, 1.0).multiply(a));
    }

    fn getRay(self: *Camera, x: u64, y: u64) Ray {
        // Get a randomly-sampled camera ray for the pixel at location x,y, originating from
        // the camera defocus disk.
        const pixel_center = self.pixel00_loc.plus(self.pixel_delta_u.multiply(@floatFromInt(x))).plus(self.pixel_delta_v.multiply(@floatFromInt(y)));
        const pixel_sample = pixel_center.plus(self.pixelSampleSquare());

        const ray_origin = if (self.defocus_angle <= 0) self.center else self.defocusDiskSample();
        const ray_direction = pixel_sample.minus(ray_origin);

        return Ray.init(ray_origin, ray_direction);
    }

    fn pixelSampleSquare(self: *Camera) Vec3 {
        const px = -0.5 * rand.randomFloat();
        const py = -0.5 * rand.randomFloat();

        return self.pixel_delta_u.multiply(px).plus(self.pixel_delta_v.multiply(py));
    }

    fn defocusDiskSample(self: *Camera) Vec3 {
        const p = vecs.randomInUnitDisk();
        return self.center.plus(self.defocus_disk_u.multiply(p.x)).plus(self.defocus_disk_v.multiply(p.y));
    }
};

const SharedStateImageWriter = struct {
    buffer: [][]ColorAndSamples,

    pub fn init(buffer: [][]ColorAndSamples) SharedStateImageWriter {
        return .{
            .buffer = buffer,
        };
    }

    pub fn writeColor(self: SharedStateImageWriter, x: u64, y: u64, color: Color, number_of_samples: u64) !void {
        self.buffer[x][y] = ColorAndSamples{
            .color = self.buffer[x][y].color.plus(color),
            .number_of_samples = number_of_samples,
        };
    }
};
