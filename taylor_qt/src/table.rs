use std::collections::HashMap;
use std::f64::consts::PI;
use lazy_static::lazy_static;

lazy_static! {
    static ref TRIG_TABLE: HashMap<&'static str, HashMap<u32, f64>> = {
        let mut table = HashMap::new();
        let mut sin_data = HashMap::new();
        let mut cos_data = HashMap::new();

        for deg in 0..=360 {
            let rad = degrees_to_radians(deg as f64);
            sin_data.insert(deg, rad.sin());
            cos_data.insert(deg, rad.cos());
        }

        table.insert("sin", sin_data);
        table.insert("cos", cos_data);
        table
    };
}

pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

pub fn get_value(func: &str, degrees: u32) -> Option<f64> {
    let normalized_deg = degrees % 360;
    TRIG_TABLE.get(func)?.get(&normalized_deg).copied()
}