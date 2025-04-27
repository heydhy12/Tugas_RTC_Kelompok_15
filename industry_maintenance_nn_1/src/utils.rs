use std::io;
use crate::model::NeuralNetwork;


pub fn get_user_input(prompt: &str) -> f64 {
    loop {
        println!("{}", prompt);
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        
        match input.trim().parse() {
            Ok(num) => return num,
            Err(_) => println!("Please enter a valid number!"),
        }
    }
}

pub fn manual_prediction(nn: &NeuralNetwork) {
    println!("\nManual Prediction Mode");
    println!("---------------------");
    
    let air_temp = get_user_input("Enter air temperature (K):");
    let process_temp = get_user_input("Enter process temperature (K):");
    let rotational_speed = get_user_input("Enter rotational speed (rpm):");
    let torque = get_user_input("Enter torque (Nm):");
    
    let input = vec![air_temp, process_temp, rotational_speed, torque];
    let (predicted_class, probabilities) = nn.predict_single(&input);
    
    println!("\nPrediction Results:");
    println!("------------------");
    println!("Most likely failure type: {}", predicted_class);
    println!("\nProbability distribution:");
    println!("Power Failure: {:.2}%", probabilities[0] * 100.0);
    println!("Overstrain Failure: {:.2}%", probabilities[1] * 100.0);
    println!("No Failure: {:.2}%", probabilities[2] * 100.0);
    println!("Heat Dissipation Failure: {:.2}%", probabilities[3] * 100.0);
}