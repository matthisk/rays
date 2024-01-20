const std = @import("std");
const Color = @import("color.zig").Color;
const Ray = @import("ray.zig").Ray;
const HitRecord = @import("objects.zig").HitRecord;
const vector = @import("vector.zig");
const rand = @import("rand.zig");

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

    pub fn init(albedo: Color) Material {
        return Material{ .lambertian = Lambertian{ .albedo = albedo } };
    }

    pub fn scatter(self: Lambertian, _: Ray, record: HitRecord, attenuation: *Color, scattered: *Ray) bool {
        var scatter_direction = record.normal + vector.randomUnitVector();

        if (vector.nearZero(scatter_direction)) {
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
        const reflected = vector.reflect(vector.unitVector(r_in.direction), record.normal);

        scattered.* = Ray{ .origin = record.p, .direction = reflected + vector.randomUnitVector() * vector.splat3(self.fuzz) };
        attenuation.* = self.albedo;

        return true;
    }
};

pub const Dielectric = struct {
    index_of_refraction: f64,

    pub fn scatter(self: Dielectric, r_in: Ray, record: HitRecord, attenuation: *Color, scattered: *Ray) bool {
        attenuation.* = Color{ 1.0, 1.0, 1.0 };
        const refraction_ratio = if (record.front_face) (1.0 / self.index_of_refraction) else self.index_of_refraction;

        const unit_direction = vector.unitVector(r_in.direction);
        const cos_theta = @min(vector.dot(-unit_direction, record.normal), 1.0);
        const sin_theta = std.math.sqrt(1.0 - cos_theta * cos_theta);

        const cannot_refract = refraction_ratio * sin_theta > 1.0;
        var direction: vector.Vector3 = undefined;

        if (cannot_refract or reflectance(cos_theta, refraction_ratio) > rand.randomFloat()) {
            direction = vector.reflect(unit_direction, record.normal);
        } else {
            direction = vector.refract(unit_direction, record.normal, refraction_ratio);
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
