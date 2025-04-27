use plotters::prelude::*;

pub fn plot_metrics(
    training_loss: &[f64],
    validation_loss: &[f64],
    training_accuracy: &[f64],
    validation_accuracy: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("metrics_plot.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(400);

    // Plot Loss
    let max_steps = training_loss.len().max(validation_loss.len());
    let max_loss = training_loss.iter()
        .chain(validation_loss.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart_loss = ChartBuilder::on(&upper)
        .caption("Training & Validation Loss", ("sans-serif", 25))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_steps, 0.0..max_loss)?;

    chart_loss.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    chart_loss.draw_series(LineSeries::new(
        training_loss.iter().enumerate().map(|(x, y)| (x, *y)),
        &RED,
    ))?.label("Training Loss")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart_loss.draw_series(LineSeries::new(
        validation_loss.iter().enumerate().map(|(x, y)| (x, *y)),
        &BLUE,
    ))?.label("Validation Loss")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart_loss.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Plot Accuracy
    let max_acc = training_accuracy.iter()
        .chain(validation_accuracy.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart_acc = ChartBuilder::on(&lower)
        .caption("Training & Validation Accuracy", ("sans-serif", 25))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_steps, 0.0..max_acc)?;

    chart_acc.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Accuracy")
        .draw()?;

    chart_acc.draw_series(LineSeries::new(
        training_accuracy.iter().enumerate().map(|(x, y)| (x, *y)),
        &GREEN,
    ))?.label("Training Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart_acc.draw_series(LineSeries::new(
        validation_accuracy.iter().enumerate().map(|(x, y)| (x, *y)),
        &MAGENTA,
    ))?.label("Validation Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));

    chart_acc.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}