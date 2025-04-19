use std::os::raw::{c_double, c_char};
use std::ffi::CStr;


#[unsafe(no_mangle)]
pub extern "C" fn rust_factorial(n: u32) -> c_double {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_taylor_series(
    func: *const c_char,
    x: c_double,
    is_degree: bool,
    result: *mut c_double,
    terms: *mut c_double,
) -> *const c_char {
    let func_str = match unsafe { CStr::from_ptr(func) }.to_str() {
        Ok(s) => s,
        Err(_) => return "Invalid string encoding\0".as_ptr() as *const c_char,
    };
    
    let x_rad = if is_degree { x.to_radians() } else { x };

    let (f_x, f1_x, f2_x, f3_x, f4_x) = match func_str {
        "sin" => (
            x_rad.sin(),
            x_rad.cos(),
            -x_rad.sin() / rust_factorial(2),
            -x_rad.cos() / rust_factorial(3),
            x_rad.sin() / rust_factorial(4),
        ),
        "cos" => (
            x_rad.cos(),
            -x_rad.sin(),
            -x_rad.cos() / rust_factorial(2),
            x_rad.sin() / rust_factorial(3),
            x_rad.cos() / rust_factorial(4),
        ),
        _ => return "Invalid function\0".as_ptr() as *const c_char,
    };

    unsafe {
        *result = f_x + f1_x * (x_rad - x_rad) + f2_x * (x_rad - x_rad).powi(2)
                 + f3_x * (x_rad - x_rad).powi(3) + f4_x * (x_rad - x_rad).powi(4);
    
        *terms = f_x;
        *terms.add(1) = f1_x;
        *terms.add(2) = f2_x;
        *terms.add(3) = f3_x;
        *terms.add(4) = f4_x;
    }

    std::ptr::null()
}