const std = @import("std");
const Color = @import("color.zig").Color;
const Ray = @import("ray.zig").Ray;
const HitRecord = @import("main.zig").HitRecord;
const vecs = @import("vec3.zig");
const rand = @import("rand.zig");

const Vec3 = vecs.Vec3;

pub const Material = union(enum) {
    lambertian: Lambertian,
    metal: Metal,
    dielectric: Dielectric,

    pub fn scatter(self: Material, ray_in: Ray, record: HitRecord, attenuation: *Color, scattered: *Ray) bool {
        return switch (self) {
            inline else => |case| case.scatter(ray_in, record, attenuation, scattered),
        };
    }
};

pub const Lambertian = struct {
    albedo: Color,

    pub fn scatter(self: Lambertian, _: Ray, record: HitRecord, attenuation: *Color, scattered: *Ray) bool {
        var scatter_direction = record.normal.plus(vecs.randomUnitVector());

        if (scatter_direction.nearZero()) {
            scatter_direction = record.normal;
        }

        scattered.* = Ray{ .origin = record.p, .direction = scatter_direction };
        attenuation.* = self.albedo;

        return true;
    }
};

pub const Metal = struct {
    albedo: Color,
    fuzz: f64,

    pub fn init(albedo: Color, f: f64) Metal {
        return Metal{
            .albedo = albedo,
            .fuzz = if (f < 1) f else 1,
        };
    }

    pub fn scatter(self: Metal, r_in: Ray, record: HitRecord, attenuation: *Color, scattered: *Ray) bool {
        const reflected = vecs.reflect(vecs.unitVector(r_in.direction), record.normal);

        scattered.* = Ray{ .origin = record.p, .direction = reflected.plus(vecs.randomUnitVector().multiply(self.fuzz)) };
        attenuation.* = self.albedo;

        return true;
    }
};

pub const Dielectric = struct {
    index_of_refraction: f64,

    pub fn scatter(self: Dielectric, r_in: Ray, record: HitRecord, attenuation: *Color, scattered: *Ray) bool {
        attenuation.* = Color.init(1.0, 1.0, 1.0);
        const refraction_ratio = if (record.front_face) (1.0 / self.index_of_refraction) else self.index_of_refraction;

        const unit_direction = vecs.unitVector(r_in.direction);
        const cos_theta = @min(unit_direction.multiply(-1).dot(record.normal), 1.0);
        const sin_theta = std.math.sqrt(1.0 - cos_theta * cos_theta);

        const cannot_refract = refraction_ratio * sin_theta > 1.0;
        var direction: Vec3 = undefined;

        if (cannot_refract or reflectance(cos_theta, refraction_ratio) > rand.randomFloat()) {
            direction = vecs.reflect(unit_direction, record.normal);
        } else {
            direction = vecs.refract(unit_direction, record.normal, refraction_ratio);
        }

        scattered.* = Ray{ .origin = record.p, .direction = direction };

        return true;
    }

    fn reflectance(cosine: f64, ref_idx: f64) f64 {
        var r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;

        return r0 + (1 - r0) * std.math.pow(f64, (1 - cosine), 5);
    }
};
