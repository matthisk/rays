const vecs = @import("vec3.zig");

pub const Ray = struct {
    origin: vecs.Vec3,
    direction: vecs.Vec3,

    pub fn init(origin: vecs.Vec3, direction: vecs.Vec3) Ray {
        return Ray{ .origin = origin, .direction = direction };
    }

    pub fn at(self: Ray, t: f64) vecs.Vec3 {
        return self.origin.plus(self.direction.multiply(t));
    }
};
