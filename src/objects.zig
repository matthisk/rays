const std = @import("std");
const math = @import("math.zig");
const Ray = @import("ray.zig").Ray;
const Interval = @import("interval.zig");
const vector = @import("vector.zig");
const Vector2 = vector.Vector2;
const Vector3 = vector.Vector3;
const Lambertian = @import("material.zig").Lambertian;
const Material = @import("material.zig").Material;
const Aabb = @import("aabb.zig");
const rand = @import("rand.zig");

pub const HitRecord = struct {
    p: Vector3 = Vector3{ 0, 0, 0 },
    normal: Vector3 = Vector3{ 0, 0, 0 },
    t: f64 = 0,
    u: f64 = 0,
    v: f64 = 0,
    front_face: bool = false,
    mat: Material = undefined,

    // Sets the hit record normal
    // NOTE: the parameter `outward_normal` is assumed to have unit length.
    pub fn setFaceNormal(self: *HitRecord, ray: Ray, outward_normal: Vector3) void {
        self.front_face = vector.dot(ray.direction, outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else -outward_normal;
    }
};

pub const Hittable = union(enum) {
    sphere: Sphere,
    planar: Planar,
    list: HittableList,
    tree: BvhTree,
    translate: Translate,
    rotate_y: RotateY,

    pub fn hit(self: Hittable, ray: Ray, ray_t: Interval) ?HitRecord {
        return switch (self) {
            inline else => |case| case.hit(ray, ray_t),
        };
    }

    pub fn bbox(self: Hittable) Aabb {
        return switch (self) {
            inline else => |case| case.bounding_box,
        };
    }
};

pub const Sphere = struct {
    center: Vector3,
    center_vec: ?Vector3 = null,
    radius: f64,
    mat: Material,
    bounding_box: Aabb,

    pub fn init(center: Vector3, radius: f64, mat: Material) Hittable {
        return Hittable{ .sphere = Sphere{ .center = center, .radius = radius, .mat = mat, .bounding_box = Aabb.init(center - vector.splat3(radius), center + vector.splat3(radius)) } };
    }

    pub fn initWithMotion(center_1: Vector3, center_2: Vector3, radius: f64, mat: Material) Hittable {
        const rvec = vector.splat3(radius);
        const aabb_1 = Aabb.init(center_1 - rvec, center_1 + rvec);
        const aabb_2 = Aabb.init(center_2 - rvec, center_2 + rvec);
        const bbox = Aabb.from(aabb_1, aabb_2);
        return Hittable{ .sphere = Sphere{ .center = center_1, .center_vec = center_2 - center_1, .radius = radius, .mat = mat, .bounding_box = bbox } };
    }

    pub fn hit(self: Sphere, ray: Ray, ray_t: Interval) ?HitRecord {
        const center = self.centerAtTime(ray.time);

        const oc = ray.origin - center;
        const a = vector.lengthSquared(ray.direction);
        const half_b = vector.dot(oc, ray.direction);
        const cc = vector.lengthSquared(oc) - self.radius * self.radius;
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
        const outward_normal = (hit_record.p - center) / vector.splat3(self.radius);
        hit_record.setFaceNormal(ray, outward_normal);

        const uv = getSphereUv(outward_normal);
        hit_record.u = uv[0];
        hit_record.v = uv[1];

        return hit_record;
    }

    fn centerAtTime(self: Sphere, time: f64) Vector3 {
        if (self.center_vec) |center_vec| {
            return self.center + vector.splat3(time) * center_vec;
        } else {
            return self.center;
        }
    }

    fn getSphereUv(p: Vector3) Vector2 {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
        //
        const theta = std.math.acos(-p[1]);
        const phi = std.math.atan2(f64, -p[2], p[0]) + std.math.pi;

        return Vector2{
            (phi / (2 * std.math.pi)),
            (theta / std.math.pi),
        };
    }
};

test "sphere hitbox" {
    const sphere = Sphere.init(Vector3{ 10, 10, 10 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));

    try std.testing.expectEqual(Aabb.init(Vector3{ 5, 5, 5 }, Vector3{ 15, 15, 15 }), sphere.bbox());
}

pub const TwoDimPrimitive = enum {
    quad,
    triangle,
    disk,
};

pub const Planar = struct {
    primitive: TwoDimPrimitive, // the 2D primitive this object represents.
    Q: Vector3, // origin point of the quad.
    u: Vector3, // vector from Q to top left of quad.
    v: Vector3, // vector from Q to bottom right of quad.
    w: Vector3, // constant vector used to transform point P into 2d coordinates.
    normal: Vector3,
    D: f64,
    mat: Material,
    bounding_box: Aabb,

    pub fn initDisk(C: Vector3, r: f64, N: Vector3, mat: Material) Hittable {
        var V = vector.cross(N, Vector3{ 0, 0, 1 });
        var U = vector.cross(N, V);
        const z_axis_aligned = vector.length(V) == 0;

        if (z_axis_aligned) {
            return Planar.init(TwoDimPrimitive.disk, C, Vector3{ 0, r, 0 }, Vector3{ r, 0, 0 }, mat);
        } else {
            U = vector.unitVector(U) * vector.splat3(r);
            V = vector.unitVector(V) * vector.splat3(r);
            return Planar.init(TwoDimPrimitive.disk, C, U, V, mat);
        }
    }

    pub fn initTriangle(Q: Vector3, u: Vector3, v: Vector3, mat: Material) Hittable {
        return Planar.init(TwoDimPrimitive.triangle, Q, u, v, mat);
    }

    pub fn initQuad(Q: Vector3, u: Vector3, v: Vector3, mat: Material) Hittable {
        return Planar.init(TwoDimPrimitive.quad, Q, u, v, mat);
    }

    pub fn init(primitive: TwoDimPrimitive, Q: Vector3, u: Vector3, v: Vector3, mat: Material) Hittable {
        const n = vector.cross(u, v);
        const normal = vector.unitVector(n);
        const D = vector.dot(normal, Q);
        const w = n / vector.splat3(vector.dot(n, n));

        return Hittable{ .planar = Planar{
            .primitive = primitive,
            .Q = Q,
            .v = v,
            .u = u,
            .mat = mat,
            .normal = normal,
            .D = D,
            .w = w,
            .bounding_box = compute_bbox(primitive, Q, u, v),
        } };
    }

    pub fn hit(self: Planar, ray: Ray, ray_t: Interval) ?HitRecord {
        const denom = vector.dot(self.normal, ray.direction);

        // No hit if the ray is parallel to the plane.
        if (@fabs(denom) < 1e-8) return null;

        const t = (self.D - vector.dot(self.normal, ray.origin)) / denom;
        // No hit if the hit point parameter t is outside of the ray.
        if (!ray_t.contains(t)) return null;

        var rec = HitRecord{};
        const intersection = ray.at(t);
        const planar_hitpt_vector = intersection - self.Q;
        const alpha = vector.dot(self.w, vector.cross(planar_hitpt_vector, self.v));
        const beta = vector.dot(self.w, vector.cross(self.u, planar_hitpt_vector));

        if (!self.isInterior(alpha, beta, &rec))
            return null;

        rec.t = t;
        rec.p = intersection;
        rec.mat = self.mat;
        rec.setFaceNormal(ray, self.normal);

        return rec;
    }

    pub fn bbox(self: Planar) Aabb {
        return self.bounding_box;
    }

    fn compute_bbox(primitive: TwoDimPrimitive, Q: Vector3, u: Vector3, v: Vector3) Aabb {
        // For a disk primitive Q points to the center of the disk, not the bottom right corner.
        // Thus we require a different calculation to construct the bounding box.
        if (primitive == .disk) {
            const aabb1 = Aabb.init(Q - u - v, Q + u + v);
            const aabb2 = Aabb.init(Q + u - v, Q - u + v);
            return Aabb.from(aabb1, aabb2);
        }
        const aabb1 = Aabb.init(Q, Q + u + v);
        const aabb2 = Aabb.init(Q + u, Q + v);
        return Aabb.from(aabb1, aabb2);
    }

    fn isInterior(self: Planar, a: f64, b: f64, rec: *HitRecord) bool {
        const unit_interval = Interval{ .min = 0, .max = 1 };

        switch (self.primitive) {
            .quad => {
                if (!unit_interval.contains(a) or !unit_interval.contains(b))
                    return false;
            },
            .triangle => {
                if (a < 0 or b < 0 or a + b > 1)
                    return false;
            },
            .disk => {
                if (@sqrt(a * a + b * b) > vector.length(self.u))
                    return false;
            },
        }

        rec.u = a;
        rec.v = b;
        return true;
    }
};

pub const Translate = struct {
    object: *const Hittable,
    offset: Vector3,
    bounding_box: Aabb,

    pub fn init(object: *const Hittable, offset: Vector3) Hittable {
        return Hittable{
            .translate = Translate{ .object = object, .offset = offset, .bounding_box = object.bbox().plus(offset) },
        };
    }

    pub fn hit(self: Translate, ray: Ray, ray_t: Interval) ?HitRecord {
        const origin = ray.origin - self.offset;
        const offset_ray = Ray.initWithTime(origin, ray.direction, ray.time);

        if (self.object.hit(offset_ray, ray_t)) |hit_record| {
            return HitRecord{
                .p = hit_record.p + self.offset,
                .normal = hit_record.normal,
                .t = hit_record.t,
                .u = hit_record.u,
                .v = hit_record.v,
                .front_face = hit_record.front_face,
                .mat = hit_record.mat,
            };
        }

        return null;
    }
};

pub const RotateY = struct {
    object: *const Hittable,
    cos_theta: f64,
    sin_theta: f64,
    bounding_box: Aabb,

    pub fn init(object: *const Hittable, degrees: f32) Hittable {
        const theta = math.degreesToRadians(degrees);
        const cos_theta = @cos(theta);
        const sin_theta = @sin(theta);

        const bbox = object.bbox();
        var min = Vector3{ -99999999, -99999999, -99999999 };
        var max = Vector3{ 99999999, 99999999, 99999999 };

        for (0..2) |i| {
            for (0..2) |j| {
                for (0..2) |k| {
                    // Iterate through all 8 corners of the cube.
                    const x = @as(f64, @floatFromInt(i)) * bbox.x.max + @as(f64, @floatFromInt(1 - i)) * bbox.x.min;
                    const y = @as(f64, @floatFromInt(j)) * bbox.y.max + @as(f64, @floatFromInt(1 - j)) * bbox.y.min;
                    const z = @as(f64, @floatFromInt(k)) * bbox.z.max + @as(f64, @floatFromInt(1 - k)) * bbox.z.min;

                    const new_x = cos_theta * x + sin_theta * z;
                    const new_z = -sin_theta * x + cos_theta * z;

                    const tester = Vector3{ new_x, y, new_z };

                    for (0..2) |c| {
                        min[c] = @min(min[c], tester[c]);
                        max[c] = @max(max[c], tester[c]);
                    }
                }
            }
        }

        return Hittable{ .rotate_y = RotateY{ .object = object, .cos_theta = @cos(theta), .sin_theta = @sin(theta), .bounding_box = Aabb.init(min, max) } };
    }

    pub fn hit(self: RotateY, ray: Ray, ray_t: Interval) ?HitRecord {
        // Translate ray into object space.
        const rotated_ray = Ray.initWithTime(self.rotate(ray.origin), self.rotate(ray.direction), ray.time);

        if (self.object.hit(rotated_ray, ray_t)) |hit_record| {
            // Change the intersection point from object space to world space.
            return HitRecord{
                .p = self.unRotate(hit_record.p),
                .normal = self.unRotate(hit_record.normal),
                .t = hit_record.t,
                .u = hit_record.u,
                .v = hit_record.v,
                .front_face = hit_record.front_face,
                .mat = hit_record.mat,
            };
        }

        return null;
    }

    pub fn rotate(self: RotateY, v: Vector3) Vector3 {
        return Vector3{
            self.cos_theta * v[0] - self.sin_theta * v[2],
            v[1],
            self.sin_theta * v[0] + self.cos_theta * v[2],
        };
    }

    pub fn unRotate(self: RotateY, v: Vector3) Vector3 {
        return Vector3{
            self.cos_theta * v[0] + self.sin_theta * v[2],
            v[1],
            -self.sin_theta * v[0] + self.cos_theta * v[2],
        };
    }
};

pub const HittableList = struct {
    objects: []const Hittable,
    bounding_box: Aabb,

    pub fn init(objects: []const Hittable) Hittable {
        var bouding_box = objects[0].bbox();
        for (objects[1..]) |obj| {
            bouding_box = Aabb.from(bouding_box, obj.bbox());
        }

        return Hittable{ .list = HittableList{
            .objects = objects,
            .bounding_box = bouding_box,
        } };
    }

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

test "hittable list bbox" {
    const sphere_1 = Sphere.init(Vector3{ 10, 10, 10 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    const sphere_2 = Sphere.init(Vector3{ 20, 20, 20 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    const sphere_3 = Sphere.init(Vector3{ 30, 30, 30 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    const sphere_4 = Sphere.init(Vector3{ 40, 40, 40 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    const hittable = HittableList.init(&[4]Hittable{ sphere_1, sphere_2, sphere_3, sphere_4 });

    // The bounding box's left bottom corner is equal to the sphere_1's bbox left bottom.
    // The bounding box's top right corner is equal to the sphere_1's bbox left bottom.
    try std.testing.expectEqual(Aabb.init(Vector3{ 5, 5, 5 }, Vector3{ 45, 45, 45 }), hittable.bbox());
}

pub const BvhTree = struct {
    allocator: std.mem.Allocator,
    root: *const BvhNode,
    bounding_box: Aabb,

    pub fn init(allocator: std.mem.Allocator, src_objects: []Hittable, start: usize, end: usize) !BvhTree {
        const root = try constructTree(allocator, src_objects, start, end);
        return BvhTree{
            .allocator = allocator,
            .root = root,
            .bounding_box = root.bbox,
        };
    }

    pub fn deinit(self: *const BvhTree) void {
        BvhNode.deinit(self.allocator, self.root);
    }

    pub fn hit(self: *const BvhTree, ray: Ray, ray_t: Interval) ?HitRecord {
        return self.root.hit(ray, ray_t);
    }

    fn constructTree(allocator: std.mem.Allocator, src_objects: []Hittable, start: usize, end: usize) !*BvhNode {
        var left: *BvhNode = undefined;
        var right: *BvhNode = undefined;

        const obj_span = end - start;

        switch (obj_span) {
            1 => {
                return makeLeaf(allocator, &src_objects[start]);
            },
            2 => {
                const comparison = switch (rand.randomIntBetween(0, 3)) {
                    0 => boxCompareX({}, src_objects[start], src_objects[start + 1]),
                    1 => boxCompareY({}, src_objects[start], src_objects[start + 1]),
                    2 => boxCompareZ({}, src_objects[start], src_objects[start + 1]),
                    else => boxCompareX({}, src_objects[start], src_objects[start + 1]),
                };

                // TODO select random axis.
                if (comparison) {
                    left = try makeLeaf(allocator, &src_objects[start]);
                    right = try makeLeaf(allocator, &src_objects[start + 1]);
                } else {
                    left = try makeLeaf(allocator, &src_objects[start + 1]);
                    right = try makeLeaf(allocator, &src_objects[start]);
                }
            },
            else => {
                const axis = rand.randomIntBetween(0, 3);
                switch (axis) {
                    0 => std.sort.heap(Hittable, src_objects[start..end], {}, boxCompareX),
                    1 => std.sort.heap(Hittable, src_objects[start..end], {}, boxCompareY),
                    2 => std.sort.heap(Hittable, src_objects[start..end], {}, boxCompareZ),
                    else => std.sort.heap(Hittable, src_objects[start..end], {}, boxCompareX),
                }

                const mid = start + obj_span / 2;
                left = try constructTree(allocator, src_objects, start, mid);
                right = try constructTree(allocator, src_objects, mid, end);
            },
        }

        return makeNode(allocator, left, right, Aabb.from(left.bbox, right.bbox));
    }

    fn makeNode(allocator: std.mem.Allocator, left: *const BvhNode, right: *const BvhNode, bounding_box: Aabb) !*BvhNode {
        const result = try allocator.create(BvhNode);
        result.left = left;
        result.right = right;
        result.leaf = null;
        result.bbox = bounding_box;
        return result;
    }

    fn makeLeaf(allocator: std.mem.Allocator, hittable: *const Hittable) !*BvhNode {
        const result = try allocator.create(BvhNode);
        result.leaf = hittable;
        result.left = null;
        result.right = null;
        result.bbox = hittable.bbox();
        return result;
    }

    fn boxCompareX(context: void, lhs: Hittable, rhs: Hittable) bool {
        _ = context;
        return lhs.bbox().x.min < rhs.bbox().x.min;
    }

    fn boxCompareY(context: void, lhs: Hittable, rhs: Hittable) bool {
        _ = context;
        return lhs.bbox().y.min < rhs.bbox().y.min;
    }

    fn boxCompareZ(context: void, lhs: Hittable, rhs: Hittable) bool {
        _ = context;
        return lhs.bbox().z.min < rhs.bbox().z.min;
    }
};

pub const BvhNode = struct {
    id: usize = 0,
    leaf: ?*const Hittable = null,
    left: ?*const BvhNode = null,
    right: ?*const BvhNode = null,
    bbox: Aabb = undefined,

    pub fn deinit(allocator: std.mem.Allocator, n: *const BvhNode) void {
        if (n.left) |node| {
            deinit(allocator, node);
        }
        if (n.right) |node| {
            deinit(allocator, node);
        }
        allocator.destroy(n);
    }

    pub fn hit(self: *const BvhNode, ray: Ray, ray_t: Interval) ?HitRecord {
        if (self.leaf) |hittable| {
            return hittable.hit(ray, ray_t);
        }

        if (!self.bbox.hit(ray, ray_t)) {
            return null;
        }

        const hit_record_l = self.left.?.hit(ray, ray_t);
        const hit_record_r = self.right.?.hit(ray, Interval{ .min = ray_t.min, .max = if (hit_record_l != null) hit_record_l.?.t else ray_t.max });

        return hit_record_r orelse hit_record_l orelse null;
    }

    pub fn print(self: *const BvhNode, allocator: std.mem.Allocator) !void {
        const Stack = std.ArrayList(*const BvhNode);

        var stack = Stack.init(allocator);
        defer stack.deinit();

        try stack.append(self);
        var level_size: usize = 0;
        var levels: usize = 0;

        while (stack.items.len > 0) {
            levels += 1;
            level_size = stack.items.len;
            std.debug.print("level_size = {d}\n", .{level_size});

            while (level_size > 0) : (level_size -= 1) {
                var node = stack.pop();

                std.debug.print("idx = {d} ", .{node.id});
                node.bbox.print();

                if (node.left) |n| {
                    try stack.insert(0, n);
                }
                if (node.right) |n| {
                    try stack.insert(0, n);
                }
            }
            std.debug.print("\n", .{});
        }

        std.debug.print("levels = {d}\n", .{levels});
    }
};

test "bvh tree" {
    const sphere_1 = Sphere.init(Vector3{ 40, 40, 40 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    const sphere_2 = Sphere.init(Vector3{ 20, 20, 20 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    const sphere_3 = Sphere.init(Vector3{ 10, 10, 10 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    const sphere_4 = Sphere.init(Vector3{ 30, 30, 30 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    var objects = [_]Hittable{ sphere_1, sphere_2, sphere_3, sphere_4 };

    const tree = try BvhTree.init(std.testing.allocator, objects[0..], 0, 4);
    defer tree.deinit();

    // level 1
    try std.testing.expectEqual(Aabb.init(Vector3{ 5, 5, 5 }, Vector3{ 25, 25, 25 }), tree.root.left.?.bbox);
    try std.testing.expectEqual(Aabb.init(Vector3{ 25, 25, 25 }, Vector3{ 45, 45, 45 }), tree.root.right.?.bbox);

    // left level 2
    try std.testing.expectEqual(Aabb.init(Vector3{ 5, 5, 5 }, Vector3{ 15, 15, 15 }), tree.root.left.?.left.?.bbox);
    try std.testing.expectEqual(Aabb.init(Vector3{ 15, 15, 15 }, Vector3{ 25, 25, 25 }), tree.root.left.?.right.?.bbox);

    // right level 2
    try std.testing.expectEqual(Aabb.init(Vector3{ 25, 25, 25 }, Vector3{ 35, 35, 35 }), tree.root.right.?.left.?.bbox);
    try std.testing.expectEqual(Aabb.init(Vector3{ 35, 35, 35 }, Vector3{ 45, 45, 45 }), tree.root.right.?.right.?.bbox);
}

test "big bvh tree" {
    const len = 347;

    var objects = [_]Hittable{undefined} ** len;
    for (0..len) |i| {
        const center: f64 = @floatFromInt(i * 10);
        objects[i] = Sphere.init(Vector3{ center, center, center }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));
    }

    const tree = try BvhTree.init(std.testing.allocator, objects[0..], 0, len);
    defer tree.deinit();

    try std.testing.expectEqual(Aabb.init(Vector3{ -5, -5, -5 }, Vector3{ 346 * 10 + 5, 346 * 10 + 5, 346 * 10 + 5 }), tree.bounding_box);

    var depth: usize = 0;
    var node: ?*const BvhNode = tree.root;
    while (node != null) : (node = node.?.left) {
        depth += 1;
    }

    try std.testing.expectEqual(@as(usize, std.math.log2(len) + 1), depth);
}

test "bvh tree benchmark" {
    const len = 10;
    const num_rays = 10_000;

    var objects = [_]Hittable{undefined} ** len;
    for (0..len) |i| {
        const center: f64 = rand.randomBetween(0, 10);
        const zaxis: f64 = 10;
        const radius: f64 = rand.randomBetween(0.2, 0.25);
        objects[i] = Sphere.init(Vector3{ center, center, zaxis }, radius, Lambertian.init(Vector3{ 0, 0, 0 }));
    }

    const list = HittableList.init(objects[0..]);
    const tree = try BvhTree.init(std.testing.allocator, objects[0..], 0, len);
    defer tree.deinit();

    var timer = try std.time.Timer.start();
    var hits: usize = 0;

    for (0..num_rays) |_| {
        const hit = tree.hit(Ray.init(Vector3{ 0, 0, 0 }, vector.random()), Interval{ .min = 0.001, .max = std.math.inf(f64) });
        if (hit) |record| {
            _ = record;
            hits += 1;
        }
    }

    const timing = timer.read();
    _ = timing;

    // std.debug.print("\nbvh timing = {d}ms hits = {d}\n", .{ timing / 1_000_000, hits });

    timer.reset();
    hits = 0;

    for (0..num_rays) |_| {
        const hit = list.hit(Ray.init(Vector3{ 0, 0, 0 }, vector.random()), Interval{ .min = 0.001, .max = std.math.inf(f64) });
        if (hit) |record| {
            _ = record;
            hits += 1;
        }
    }

    const timing_list = timer.read();
    _ = timing_list;

    // std.debug.print("list timing = {d}ms hits = {d}\n", .{ timing_list / 1_000_000, hits });
}
