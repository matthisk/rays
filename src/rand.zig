const std = @import("std");

var rnd = std.rand.DefaultPrng.init(0);

pub fn random_float() f32 {
    return rnd.random().float(f32);
}

pub fn random_between(min: f32, max: f32) f32 {
    return min + (max - min) * random_float();
}
