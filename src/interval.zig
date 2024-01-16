const std = @import("std");
const Interval = @This();

min: f64 = std.math.inf(f64),
max: f64 = -std.math.inf(f64),

pub fn empty() Interval {
    return Interval{ .min = std.math.inf(f64), .max = -std.math.inf(f64) };
}

pub fn universe() Interval {
    return Interval{ .min = -std.math.inf(f64), .max = std.math.inf(f64) };
}

pub fn contains(self: Interval, x: f64) bool {
    return self.min <= x and x <= self.max;
}

pub fn surrounds(self: Interval, x: f64) bool {
    return self.min < x and x < self.max;
}

pub fn clamp(self: Interval, x: f64) f64 {
    if (x < self.min) return self.min;
    if (x > self.max) return self.max;
    return x;
}
