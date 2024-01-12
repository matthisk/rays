const std = @import("std");

var rnd = std.rand.DefaultPrng.init(0);

pub fn randomFloat() f64 {
    return rnd.random().float(f64);
}

pub fn randomBetween(min: f64, max: f64) f64 {
    return min + (max - min) * randomFloat();
}
