use std::f64::consts::PI;

pub fn approximate(func: &str, x: f64) -> f64 {
    let x = x % (2.0 * PI);  // Normalisasi sudut
    match func {
        "sin" => x - x.powi(3)/6.0 + x.powi(5)/120.0 - x.powi(7)/5040.0,
        "cos" => 1.0 - x.powi(2)/2.0 + x.powi(4)/24.0 - x.powi(6)/720.0,
        _ => 0.0,
    }
}