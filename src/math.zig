const std = @import("std");

const infinity = std.math.inf(f64);
const pi = std.math.pi;

pub fn degreesToRadians(degrees: f64) f64 {
    return degrees * pi / 180;
}

pub fn linearToGamma(linear_component: f64) f64 {
    return std.math.sqrt(linear_component);
}
