const std = @import("std");
const rand = @import("rand.zig");
const vector = @import("vector.zig");

const Vector3 = vector.Vector3;
const Vector2 = vector.Vector2;

const Perlin = @This();

const point_count: usize = 256;

rand_vector: []Vector3 = undefined,
perm_x: []usize = undefined,
perm_y: []usize = undefined,
perm_z: []usize = undefined,

pub fn init(allocator: std.mem.Allocator) !*Perlin {
    const self = try allocator.create(Perlin);
    self.* = Perlin{};

    self.rand_vector = try allocator.alloc(Vector3, point_count);

    for (0..point_count) |i| {
        self.rand_vector[i] = vector.randomBetween(-1, 1);
    }

    self.perm_x = try allocator.alloc(usize, point_count);
    self.perm_y = try allocator.alloc(usize, point_count);
    self.perm_z = try allocator.alloc(usize, point_count);

    generatePerm(self.perm_x);
    generatePerm(self.perm_y);
    generatePerm(self.perm_z);

    return self;
}

pub fn destroy(self: *Perlin, allocator: std.mem.Allocator) void {
    allocator.free(self.rand_vector);
    allocator.free(self.perm_x);
    allocator.free(self.perm_y);
    allocator.free(self.perm_z);
    allocator.destroy(self);
}

pub fn noise(self: *Perlin, p: Vector3) f64 {
    var u = @fabs(p[0]) - @floor(@fabs(p[0]));
    var v = @fabs(p[1]) - @floor(@fabs(p[1]));
    var w = @fabs(p[2]) - @floor(@fabs(p[2]));

    const i: usize = @intFromFloat(@floor(@fabs(p[0])));
    const j: usize = @intFromFloat(@floor(@fabs(p[1])));
    const k: usize = @intFromFloat(@floor(@fabs(p[2])));

    var c: [2][2][2]Vector3 = .{};

    for (0..2) |di| {
        for (0..2) |dj| {
            for (0..2) |dk| {
                c[di][dj][dk] = self.rand_vector[self.perm_x[(i + di) & 255] ^ self.perm_y[(j + dj) & 255] ^ self.perm_z[(k + dk) & 255]];
            }
        }
    }

    return perlinInterp(c, u, v, w);
}

pub fn turbulence(self: *Perlin, p: Vector3, depth: usize) f64 {
    var accum: f64 = 0.0;
    var temp_p = p;
    var weight: f32 = 1.0;

    var i: usize = 0;
    while (i < depth) {
        accum += weight * self.noise(temp_p);
        weight *= 0.5;
        temp_p *= vector.splat3(2);
        i += 1;
    }

    return @fabs(accum);
}

fn perlinInterp(c: [2][2][2]Vector3, u: f64, v: f64, w: f64) f64 {
    const uu = u * u * (3 - 2 * u);
    const vv = v * v * (3 - 2 * v);
    const ww = w * w * (3 - 2 * w);
    var accum: f64 = 0;

    for (0..2) |ii| {
        for (0..2) |jj| {
            for (0..2) |kk| {
                const i: f64 = @floatFromInt(ii);
                const j: f64 = @floatFromInt(jj);
                const k: f64 = @floatFromInt(kk);

                const weight_v = Vector3{ u - i, v - j, w - k };
                accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) * (k * ww + (1 - k) * (1 - ww)) * vector.dot(c[ii][jj][kk], weight_v);
            }
        }
    }

    return accum;
}

fn trilinearInterp(c: [2][2][2]f64, u: f64, v: f64, w: f64) f64 {
    var accum: f64 = 0.0;

    for (0..2) |ii| {
        for (0..2) |jj| {
            for (0..2) |kk| {
                const i: f64 = @floatFromInt(ii);
                const j: f64 = @floatFromInt(jj);
                const k: f64 = @floatFromInt(kk);

                accum += (i * u + (1 - i) * (1 - u)) * (j * v + (1 - j) * (1 - v)) * (k * w + (1 - k) * (1 - w)) * c[ii][jj][kk];
            }
        }
    }

    return accum;
}

fn generatePerm(p: []usize) void {
    for (0..p.len) |i| {
        p[i] = i;
    }

    permute(p, point_count);
}

fn permute(p: []usize, n: usize) void {
    var i = (n - 1);

    while (i > 0) {
        const target = rand.randomIntBetween(0, i);
        const tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;

        i -= 1;
    }
}

test "Perlin" {
    const perlin = try Perlin.init(std.testing.allocator);
    defer perlin.destroy(std.testing.allocator);

    const n = perlin.noise(Vector3{ 0, 0, 0 });

    try std.testing.expectApproxEqRel(n, 0.883107, 0.1);
}
