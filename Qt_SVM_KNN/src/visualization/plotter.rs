use ndarray::{Array1, Array2};
use plotters::prelude::*;

pub fn plot_svm_neighbors(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    test_label: usize,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let caption = format!(
        "SVM Neighbors (Predicted: {})",
        match test_label {
            0 => "No Failure",
            1 => "Heat Dissipation Failure",
            2 => "Overstrain Failure",
            3 => "Power Failure",
            _ => "Unknown",
        }
    );

    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = train.column(0).iter().chain(test.column(0).iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = train.column(0).iter().chain(test.column(0).iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = train.column(1).iter().chain(test.column(1).iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = train.column(1).iter().chain(test.column(1).iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(caption.as_str(), ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Feature 1")
        .y_desc("Feature 2")
        .draw()?;

    // Draw training points
    for (i, label) in train_labels.iter().enumerate() {
        let color = match label {
            0 => GREEN,
            1 => RED,
            2 => BLUE,
            3 => YELLOW,
            _ => BLACK,
        };
        chart.draw_series(PointSeries::of_element(
            vec![(train[[i, 0]], train[[i, 1]])],
            5,
            ShapeStyle::from(&color).filled(),
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;
    }

    // Draw test point
    chart.draw_series(PointSeries::of_element(
        vec![(test[[0, 0]], test[[0, 1]])],
        10,
        ShapeStyle::from(&BLACK).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    root.present()?;
    Ok(())
}

pub fn plot_knn_neighbors(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    test_label: usize,
    neighbors: &[usize],
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let caption = format!(
        "KNN Neighbors (Predicted: {}, k={})",
        match test_label {
            0 => "No Failure",
            1 => "Heat Dissipation Failure",
            2 => "Overstrain Failure",
            3 => "Power Failure",
            _ => "Unknown",
        },
        neighbors.len()
    );

    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = train.column(0).iter().chain(test.column(0).iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = train.column(0).iter().chain(test.column(0).iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = train.column(1).iter().chain(test.column(1).iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = train.column(1).iter().chain(test.column(1).iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(caption.as_str(), ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Feature 1")
        .y_desc("Feature 2")
        .draw()?;

    // Draw all training points (faint)
    for (i, label) in train_labels.iter().enumerate() {
        let color = match label {
            0 => GREEN.mix(0.2),
            1 => RED.mix(0.2),
            2 => BLUE.mix(0.2),
            3 => YELLOW.mix(0.2),
            _ => BLACK.mix(0.2),
        };
        chart.draw_series(PointSeries::of_element(
            vec![(train[[i, 0]], train[[i, 1]])],
            3,
            ShapeStyle::from(&color).filled(),
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;
    }

    // Draw nearest neighbors (bold)
    for &idx in neighbors {
        let label = train_labels[idx];
        let color = match label {
            0 => GREEN,
            1 => RED,
            2 => BLUE,
            3 => YELLOW,
            _ => BLACK,
        };
        chart.draw_series(PointSeries::of_element(
            vec![(train[[idx, 0]], train[[idx, 1]])],
            6,
            ShapeStyle::from(&color).filled(),
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;

        // Draw connecting lines
        chart.draw_series(LineSeries::new(
            vec![
                (test[[0, 0]], test[[0, 1]]),
                (train[[idx, 0]], train[[idx, 1]]),
            ],
            &BLACK.mix(0.9),
        ))?;
    }

    // Draw test point (large black)
    chart.draw_series(PointSeries::of_element(
        vec![(test[[0, 0]], test[[0, 1]])],
        9,
        ShapeStyle::from(&BLACK).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    root.present()?;
    Ok(())
}