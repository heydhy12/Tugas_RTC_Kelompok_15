use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

pub fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn knn_predict(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    k: usize,
) -> Array1<usize> {
    let (predictions, _) = knn_predict_with_neighbors(train, train_labels, test, k);
    predictions
}

pub fn knn_predict_with_neighbors(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    k: usize,
) -> (Array1<usize>, Vec<Vec<usize>>) {
    let mut predictions = Array1::zeros(test.nrows());
    let mut neighbors_list = Vec::with_capacity(test.nrows());

    for (i, test_sample) in test.axis_iter(Axis(0)).enumerate() {
        let mut distances: Vec<(usize, f64)> = train.axis_iter(Axis(0))
            .enumerate()
            .map(|(idx, train_sample)| (idx, euclidean_distance(&train_sample.to_owned(), &test_sample.to_owned())))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let nearest_indices: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
        neighbors_list.push(nearest_indices.clone());

        let mut label_counts = HashMap::new();
        for &idx in &nearest_indices {
            let label = train_labels[idx];
            *label_counts.entry(label).or_insert(0) += 1;
        }

        predictions[i] = label_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or(0);
    }

    (predictions, neighbors_list)
}

pub fn evaluate_accuracy(
    predictions: &Array1<usize>,
    actual: &Array1<usize>
) -> f32 {
    predictions.iter()
        .zip(actual.iter())
        .filter(|(pred, true_val)| pred == true_val)
        .count() as f32 / predictions.len() as f32
}