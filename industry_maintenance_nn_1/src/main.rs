mod data;
mod model;
mod plot;
mod utils;

use std::error::Error;
use std::time::Instant;
use chrono::Duration;
use std::io;

use data::load_data;
use model::NeuralNetwork;
use plot::plot_metrics;
use utils::manual_prediction;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Reading Files CSV...");
    let file_path = "csv/industry_maintenance.csv";
    let (x, y) = load_data(file_path)?;
    println!("Successfully Read {} records.", x.shape()[0]);

    //PARAMETER TRAINING
    println!("Inisialisasi neural network...");
    let input_size = 4;
    let hidden_size1 = 5;
    let hidden_size2 = 3;
    let output_size = 4;
    let learning_rate = 0.001;
    let epochs = 1000;

    let mut nn = NeuralNetwork::new(input_size, hidden_size1, hidden_size2, output_size);

    println!("\nStarting Training...");
    let start_time = Instant::now();
    let (train_loss, val_loss, train_accuracy, val_accuracy) = nn.train(&x, &y, learning_rate, epochs);
    let duration = Duration::from_std(start_time.elapsed()).unwrap();
    println!("Training Completed In {} second!", duration.num_seconds());
    println!("\nTraining Accuracy: {:.2}%", train_accuracy.last().unwrap() * 100.0);
    println!("Validation Accuracy: {:.2}%", val_accuracy.last().unwrap() * 100.0);
    
    // Save the trained model
    let model_path = "Trained_model.bin";
    nn.save(model_path)?;
    println!("\nModel successfully saved as: {}", model_path);

    // Load the trained model to demonstrate usage of `load`
    let loaded_nn = NeuralNetwork::load(model_path)?;
    println!("MModel successfully reloaded from: {}", model_path);

    println!("Creating training progress plot...");
    plot_metrics(&train_loss, &val_loss, &train_accuracy, &val_accuracy)?;
    println!("Plot successfully saved as metrics_plot.png");

    println!("\nEvaluasi Model:");
    let (train_loss, train_accuracy) = loaded_nn.evaluate(&x, &y);
    println!("Training Data - Loss: {:.4}, Akurasi: {:.2}%", train_loss, train_accuracy * 100.0);

    // Manual prediction loop
    loop {
        println!("\nMenu:");
        println!("1. Perform manual prediction");
        println!("2. Exit");
        
        let mut choice = String::new();
        io::stdin().read_line(&mut choice).expect("Failed to read line");
        
        match choice.trim() {
            "1" => manual_prediction(&loaded_nn),
            "2" => break,
            _ => println!("Invalid selection!"),
        }
    }

    Ok(())
}