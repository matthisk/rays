const std = @import("std");
const vecs = @import("vec3.zig");
const rays = @import("ray.zig");
const colors = @import("color.zig");

const Vec3 = vecs.Vec3;
const Ray = rays.Ray;
const Color = colors.Color;

const infinity = std.math.inf(f32);
const pi = std.math.pi;
var rnd = std.rand.DefaultPrng.init(0);

fn degrees_to_radians(degrees: f32) f32 {
    return degrees * pi / 180;
}

fn random_float() f32 {
    return rnd.random().float(f32);
}

fn random_between(min: f32, max: f32) f32 {
    return min + (max - min) * random_float();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const objects = [2]Hittable{
        Hittable{ .sphere = Sphere{ .center = Vec3.init(0, 0, -1), .radius = 0.5 } },
        Hittable{ .sphere = Sphere{ .center = Vec3.init(0, -100.5, -1), .radius = 100 } },
    };
    const world = Hittable{ .list = HittableList{ .objects = objects[0..] } };

    const camera = try allocator.create(Camera);
    defer allocator.destroy(camera);

    camera.* = Camera{
        .aspect_ratio = 16.0 / 9.0,
        .img_width = 400,
    };

    try camera.render(world);
}

pub fn rayColor(world: Hittable, ray: Ray) colors.Color {
    const opt_hit_record = world.hit(ray, Interval{ .min = 0, .max = std.math.inf(f32) });

    if (opt_hit_record) |hit_record| {
        return Color.init(hit_record.normal.x + 1, hit_record.normal.y + 1, hit_record.normal.z + 1).multiply(0.5);
    }

    const unit_direction = vecs.unit_vector(ray.direction);
    const a = 0.5 * (unit_direction.y + 1.0);

    return Color.init(1.0, 1.0, 1.0).multiply(1.0 - a).plus(Color.init(0.5, 0.7, 1.0).multiply(a));
}

const HitRecord = struct {
    p: Vec3 = Vec3.init(0, 0, 0),
    normal: Vec3 = Vec3.init(0, 0, 0),
    t: f32 = 0,
    front_face: bool = false,

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
    radius: f32,

    pub fn hit(self: Sphere, ray: Ray, ray_t: Interval) ?HitRecord {
        const oc = ray.origin.minus(self.center);
        const a = ray.direction.length_squared();
        const half_b = oc.dot(ray.direction);
        const c = oc.length_squared() - self.radius * self.radius;
        const discriminant = half_b * half_b - a * c;
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

const Interval = struct {
    min: f32 = std.math.inf(f32),
    max: f32 = -std.math.inf(f32),

    pub fn empty() Interval {
        return Interval{ .min = std.math.inf(f32), .max = -std.math.inf(f32) };
    }

    pub fn universe() Interval {
        return Interval{ .min = -std.math.inf(f32), .max = std.math.inf(f32) };
    }

    pub fn contains(self: Interval, x: f32) bool {
        return self.min <= x and x <= self.max;
    }

    pub fn surrounds(self: Interval, x: f32) bool {
        return self.min < x and x < self.max;
    }

    pub fn clamp(self: Interval, x: f32) f32 {
        if (x < self.min) return self.min;
        if (x > self.max) return self.max;
        return x;
    }
};

const Camera = struct {
    aspect_ratio: f32 = 1.0, // Ratio of image width over height
    img_width: u32 = 0, // Rendered image width in pixel count
    samples_per_pixel: u32 = 100, // Count of random samples for each pixel
    img_height: u32 = 0,
    center: Vec3 = Vec3.init(0, 0, 0),
    pixel00_loc: Vec3 = Vec3.init(0, 0, 0),
    pixel_delta_u: Vec3 = Vec3.init(0, 0, 0),
    pixel_delta_v: Vec3 = Vec3.init(0, 0, 0),

    pub fn render(self: *Camera, world: Hittable) std.fs.File.Writer.Error!void {
        self.initialize();

        var stdout = std.io.getStdOut().writer();
        try stdout.print("P3\n{d} {d}\n255\n", .{ self.img_width, self.img_height });

        for (1..self.img_height + 1) |y| {
            for (1..self.img_width + 1) |x| {
                var color = Color.init(0, 0, 0);

                for (1..self.samples_per_pixel + 1) |_| {
                    const ray = self.get_ray(x, y);
                    color = color.plus(rayColor(world, ray));
                }

                try self.writeColor(stdout, color, self.samples_per_pixel);
            }
        }
    }

    fn initialize(self: *Camera) void {
        const w: f32 = @floatFromInt(self.img_width);
        self.img_height = @intFromFloat(w / self.aspect_ratio);

        // Camera
        const focal_length = 1.0;
        const viewport_height = 2.0;
        const viewport_aspect_ratio = @as(f32, @floatFromInt(self.img_width)) / @as(f32, @floatFromInt(self.img_height));
        const viewport_width = viewport_height * viewport_aspect_ratio;
        self.center = Vec3.init(0, 0, 0);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        const viewport_u = Vec3.init(viewport_width, 0, 0);
        const viewport_v = Vec3.init(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.pixel_delta_u = viewport_u.divide(@floatFromInt(self.img_width));
        self.pixel_delta_v = viewport_v.divide(@floatFromInt(self.img_height));

        // Calculate the location of the upper left pixel.
        const viewport_upper_left = self.center.minus(Vec3.init(0, 0, focal_length)).minus(viewport_u.divide(2)).minus(viewport_v.divide(2));
        self.pixel00_loc = viewport_upper_left.plus(self.pixel_delta_u.plus(self.pixel_delta_v).multiply(0.5));
    }

    fn ray_color(ray: Ray, world: Hittable) Color {
        const opt_hit_record = world.hit(ray, Interval{ .min = 0, .max = std.math.inf(f32) });

        if (opt_hit_record) |hit_record| {
            return Color.init(hit_record.normal.x + 1, hit_record.normal.y + 1, hit_record.normal.z + 1).multiply(0.5);
        }

        const unit_direction = vecs.unit_vector(ray.direction);
        const a = 0.5 * (unit_direction.y + 1.0);

        return Color.init(1.0, 1.0, 1.0).multiply(1.0 - a).plus(Color.init(0.5, 0.7, 1.0).multiply(a));
    }

    fn get_ray(self: *Camera, x: u64, y: u64) Ray {
        const pixel_center = self.pixel00_loc.plus(self.pixel_delta_u.multiply(@floatFromInt(x))).plus(self.pixel_delta_v.multiply(@floatFromInt(y)));
        const pixel_sample = pixel_center.plus(self.pixel_sample_square());

        const ray_origin = self.center;
        const ray_direction = pixel_sample.minus(self.center);

        return Ray.init(ray_origin, ray_direction);
    }

    fn pixel_sample_square(self: *Camera) Vec3 {
        const px = -0.5 * random_float();
        const py = -0.5 * random_float();

        return self.pixel_delta_u.multiply(px).plus(self.pixel_delta_v.multiply(py));
    }

    fn writeColor(self: *Camera, writer: std.fs.File.Writer, color: Color, samples_per_pixel: u32) std.fs.File.Writer.Error!void {
        _ = self;

        var r = color.x;
        var g = color.y;
        var b = color.z;

        const scale = 1.0 / @as(f32, @floatFromInt(samples_per_pixel));
        r *= scale;
        g *= scale;
        b *= scale;

        const intensity = Interval{ .min = 0.0, .max = 0.999 };
        try writer.print("{d} {d} {d}\n", .{
            @floor(intensity.clamp(r) * 255.999),
            @floor(intensity.clamp(g) * 255.999),
            @floor(intensity.clamp(b) * 255.999),
        });
    }
};
