use ndarray::{Array, ArrayView2, Axis, Ix2};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray2};
use pyo3::prelude::*;

/// Apply LoRA transformation to a tensor
/// Formula: W' = W + alpha * (BA)
/// Returns the modified weight tensor
#[pyfunction]
fn apply_lora<'py>(
    py: Python<'py>,
    weight: PyReadonlyArray2<'py, f32>,
    lora_a: PyReadonlyArray2<'py, f32>,
    lora_b: PyReadonlyArray2<'py, f32>,
    alpha: f32,
) -> PyResult<&'py PyArray<f32, Ix2>> {
    // Convert to ndarray views
    let weight_view = weight.as_array();
    let lora_a_view = lora_a.as_array();
    let lora_b_view = lora_b.as_array();
    
    // Get dimensions
    let weight_shape = weight_view.shape();
    let a_shape = lora_a_view.shape();
    let b_shape = lora_b_view.shape();
    
    // Print dimensions for debugging
    println!("Weight shape: [{}, {}]", weight_shape[0], weight_shape[1]);
    println!("LoRA A shape: [{}, {}]", a_shape[0], a_shape[1]);
    println!("LoRA B shape: [{}, {}]", b_shape[0], b_shape[1]);
    
    // Basic validation
    if b_shape[1] != a_shape[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Incompatible shapes for LoRA matrices: B [{}, {}] and A [{}, {}]",
                    b_shape[0], b_shape[1], a_shape[0], a_shape[1])
        ));
    }
    
    if b_shape[0] != weight_shape[0] || a_shape[1] != weight_shape[1] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Incompatible shapes: LoRA adaptation [{}, {}] doesn't match weight [{}, {}]",
                   b_shape[0], a_shape[1], weight_shape[0], weight_shape[1])
        ));
    }
    
    // Compute BA
    let ba_product = compute_ba_product(&lora_b_view, &lora_a_view)?;
    
    // Clone weight to create a mutable copy
    let mut result = weight_view.to_owned();
    
    // Add LoRA update: W + alpha * (BA)
    for i in 0..weight_shape[0] {
        for j in 0..weight_shape[1] {
            result[[i, j]] += alpha * ba_product[[i, j]];
        }
    }
    
    // Convert back to Python array
    Ok(result.into_pyarray(py))
}

/// Compute BA matrix product
fn compute_ba_product(
    b: &ArrayView2<f32>,
    a: &ArrayView2<f32>,
) -> PyResult<Array<f32, Ix2>> {
    // We want to compute B @ A
    let b_shape = b.shape();
    let a_shape = a.shape();
    
    if b_shape[1] != a_shape[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Cannot multiply B [{}, {}] with A [{}, {}]", 
                    b_shape[0], b_shape[1], a_shape[0], a_shape[1])
        ));
    }
    
    // Create result array
    let mut result = Array::zeros((b_shape[0], a_shape[1]));
    
    // Perform matrix multiplication
    for i in 0..b_shape[0] {
        for j in 0..a_shape[1] {
            let mut sum = 0.0;
            for k in 0..b_shape[1] {
                sum += b[[i, k]] * a[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }
    
    Ok(result)
}

/// Python module definition
#[pymodule]
fn lora_ops(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_lora, m)?)?;
    Ok(())
} 