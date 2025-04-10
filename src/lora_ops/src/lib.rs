use ndarray::{Array, ArrayView2, Axis, Ix2};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

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
    // Shape validation
    if b.shape()[1] != a.shape()[0] {
        return Err(PyErr::new::<PyValueError, _>(
            format!(
                "Cannot multiply B [{}x{}] with A [{}x{}]",
                b.shape()[0], b.shape()[1],
                a.shape()[0], a.shape()[1]
            ),
        ));
    }

    // Perform matrix multiplication using `.dot()`
    let result = b.dot(a);

    Ok(result)
}

/// Add batch processing capability for multiple LoRA applications
#[pyfunction]
fn apply_lora_batch<'py>(
    py: Python<'py>,
    weight: PyReadonlyArray2<'py, f32>,
    lora_pairs: Vec<(PyReadonlyArray2<'py, f32>, PyReadonlyArray2<'py, f32>, f32)>,
) -> PyResult<&'py PyArray<f32, Ix2>> {
    // Convert to ndarray view
    let weight_view = weight.as_array();
    
    // Clone weight to create a mutable copy
    let mut result = weight_view.to_owned();
    
    // Apply each LoRA pair
    for (lora_a, lora_b, alpha) in lora_pairs {
        let lora_a_view = lora_a.as_array();
        let lora_b_view = lora_b.as_array();
        
        // Basic validation
        // Store the shape dimensions to avoid borrowing issues
        let rows = result.shape()[0];
        let cols = result.shape()[1];
        let a_shape = lora_a_view.shape();
        let b_shape = lora_b_view.shape();
        
        if b_shape[1] != a_shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Incompatible shapes for LoRA matrices: B [{}, {}] and A [{}, {}]",
                        b_shape[0], b_shape[1], a_shape[0], a_shape[1])
            ));
        }
        
        if b_shape[0] != rows || a_shape[1] != cols {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Incompatible shapes: LoRA adaptation [{}, {}] doesn't match weight [{}, {}]",
                       b_shape[0], a_shape[1], rows, cols)
            ));
        }
        
        // Compute BA
        let ba_product = compute_ba_product(&lora_b_view, &lora_a_view)?;
        
        // Add LoRA update: W + alpha * (BA)
        for i in 0..rows {
            for j in 0..cols {
                result[[i, j]] += alpha * ba_product[[i, j]];
            }
        }
    }
    
    // Convert back to Python array
    Ok(result.into_pyarray(py))
}

/// Python module definition
#[pymodule]
fn lora_ops(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_lora, m)?)?;
    m.add_function(wrap_pyfunction!(apply_lora_batch, m)?)?;
    Ok(())
} 