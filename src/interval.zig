const std = @import("std");
const Interval = @This();

min: f64 = std.math.inf(f64),
max: f64 = -std.math.inf(f64),

pub fn empty() Interval {
    return Interval{ .min = std.math.inf(f64), .max = -std.math.inf(f64) };
}

pub fn from(interval_1: Interval, interval_2: Interval) Interval {
    return Interval{
        .min = @min(interval_1.min, interval_2.min),
        .max = @max(interval_1.max, interval_2.max),
    };
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

pub fn size(self: Interval) f64 {
    return self.max - self.min;
}

pub fn expand(self: Interval, delta: f64) f64 {
    const padding = delta / 2;
    return Interval{
        .min = self.min - padding,
        .max = self.max + padding,
    };
}
