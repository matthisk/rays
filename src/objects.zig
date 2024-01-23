const std = @import("std");
const Ray = @import("ray.zig").Ray;
const Interval = @import("interval.zig");
const vector = @import("vector.zig");
const Vector3 = vector.Vector3;
const Lambertian = @import("material.zig").Lambertian;
const Material = @import("material.zig").Material;
const Aabb = @import("aabb.zig");
const rand = @import("rand.zig");

pub const HitRecord = struct {
    p: Vector3 = Vector3{ 0, 0, 0 },
    normal: Vector3 = Vector3{ 0, 0, 0 },
    t: f64 = 0,
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
    list: HittableList,
    tree: BvhTree,

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
    radius: f64,
    mat: Material,
    bounding_box: Aabb,

    pub fn init(center: Vector3, radius: f64, mat: Material) Hittable {
        return Hittable{ .sphere = Sphere{ .center = center, .radius = radius, .mat = mat, .bounding_box = Aabb.init(center - vector.splat3(radius), center + vector.splat3(radius)) } };
    }

    pub fn hit(self: Sphere, ray: Ray, ray_t: Interval) ?HitRecord {
        const oc = ray.origin - self.center;
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
        const outward_normal = (hit_record.p - self.center) / vector.splat3(self.radius);
        hit_record.setFaceNormal(ray, outward_normal);

        return hit_record;
    }
};

test "sphere hitbox" {
    const sphere = Sphere.init(Vector3{ 10, 10, 10 }, 5, Lambertian.init(Vector3{ 0, 0, 0 }));

    try std.testing.expectEqual(Aabb.init(Vector3{ 5, 5, 5 }, Vector3{ 15, 15, 15 }), sphere.bbox());
}

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
    const len = 10_000;
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

    std.debug.print("\nbvh timing = {d}ms hits = {d}\n", .{ timing / 1_000_000, hits });

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

    std.debug.print("list timing = {d}ms hits = {d}\n", .{ timing_list / 1_000_000, hits });
}
