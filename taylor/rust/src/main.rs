use std::io;

/// 1) Menghitung faktorial
fn factorial(n: u32) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

/// 2) Deret Taylor sin(x) atau cos(x)
fn taylor_series(func: &str, x: f64) {
    let a = x; 
    let h = x - a; 

    let f_x;
    let f1_x;
    let f2_x;
    let f3_x;
    let f4_x;

    match func {
        "sin" => {
            f_x = a.sin();
            f1_x = a.cos();
            f2_x = -a.sin() / factorial(2);
            f3_x = -a.cos() / factorial(3);
            f4_x = a.sin() / factorial(4);
        }
        "cos" => {
            f_x = a.cos();
            f1_x = -a.sin();
            f2_x = -a.cos() / factorial(2);
            f3_x = a.sin() / factorial(3);
            f4_x = a.cos() / factorial(4);
        }
        _ => {
            println!("Fungsi tidak dikenali! Gunakan 'sin' atau 'cos'.");
            return;
        }
    }

    let taylor_approx = f_x + f1_x * h + f2_x * h.powi(2) + f3_x * h.powi(3) + f4_x * h.powi(4);

    println!("\nDeret Taylor hingga orde ke-4 untuk {}(x) di sekitar x = {}:", func, x);
    println!("f(x)      = {:.4}", f_x);
    println!("f'(x)     = {:.4} (h)", f1_x);
    println!("f''(x)    = {:.4} (h^2)", f2_x);
    println!("f'''(x)   = {:.4} (h^3)", f3_x);
    println!("f''''(x)  = {:.4} (h^4)", f4_x);

    println!("\nDeret Taylor hingga orde ke-4:");
    println!(
        "f(x) ≈ {:.4} + {:.4}h {:.4}h^2 {:.4}h^3 {:.4}h^4",
        f_x, f1_x, f2_x, f3_x, f4_x
    );
    println!(
        "Perkiraan nilai f(x) berdasarkan deret Taylor: {:.4}",
        taylor_approx
    );
}

fn main() {
    //3) Input fungsi, Sudut, dan Nilai x
    let mut input_func = String::new();
    let mut input_angle_type = String::new();
    let mut input_x = String::new();

    println!("Masukkan fungsi (sin/cos): ");
    io::stdin().read_line(&mut input_func).expect("Gagal membaca input");
    let func = input_func.trim().to_lowercase();

    //4) Jenis Sudut
    println!("Masukkan jenis sudut (radian/derajat): ");
    io::stdin().read_line(&mut input_angle_type).expect("Gagal membaca input");
    let angle_type = input_angle_type.trim().to_lowercase();

    //5) Input nilai x
    println!("Masukkan nilai x: ");
    io::stdin().read_line(&mut input_x).expect("Gagal membaca input");
    let x: f64 = input_x.trim().parse().expect("Harap masukkan angka!");

    let x_rad = if angle_type == "derajat" {
        x.to_radians()
    } else {
        x
    };

    taylor_series(&func, x_rad);
}