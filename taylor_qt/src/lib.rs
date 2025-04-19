mod table;
mod taylor;

use std::os::raw::c_char;
use std::ffi::{CStr, CString};

#[unsafe(no_mangle)]
pub extern "C" fn calculate(
    func: *const c_char,
    angle_type: *const c_char,
    x: f64,
) -> *mut c_char {
    // Operasi unsafe harus dalam block unsafe
    let func_str = unsafe { CStr::from_ptr(func).to_str().unwrap() };
    let angle_str = unsafe { CStr::from_ptr(angle_type).to_str().unwrap() };

    let (x_rad, degrees) = if angle_str == "derajat" {
        let deg = x.round() as u32;
        (table::degrees_to_radians(x), deg)
    } else {
        let deg = (x * 180.0 / std::f64::consts::PI).round() as u32;
        (x, deg)
    };

    let taylor_val = taylor::approximate(func_str, x_rad);
    let table_val = table::get_value(func_str, degrees);

    let output = match table_val {
        Some(t_val) => format!(
            "Taylor: {:.6}\nLookup: {:.6}\nError: {:.6e}",
            taylor_val, t_val, (taylor_val - t_val).abs()
        ),
        None => format!("Taylor: {:.6}\n(Lookup tidak tersedia)", taylor_val),
    };

    // Konversi ke CString dan transfer ownership
    CString::new(output).unwrap().into_raw()
}

#[unsafe(no_mangle)]
pub extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            // Mengambil ownership kembali untuk di-free
            let _ = CString::from_raw(s);
        }
    }
}