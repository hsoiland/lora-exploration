//! # LoRA Operations for Diffusion Models
//! 
//! This module provides Rust-accelerated implementations of Low-Rank Adaptation (LoRA)
//! operations for training and inference with diffusion models.
//!
//! LoRA is a technique that enables efficient fine-tuning of large models by adding
//! low-rank decomposition matrices to existing weights, drastically reducing the number
//! of trainable parameters.
//!
//! ## Key Components
//!
//! - `LoraTrainingContext`: Manages the training state for a single layer's LoRA adaptation
//! - `AdamParams`: Implements the Adam optimizer for LoRA parameters
//! - Utility functions for applying LoRA transformations during inference
//!
//! ## Formula
//!
//! LoRA uses the following transformation: W' = W + α(BA), where:
//! - W is the original weight matrix
//! - A is a low-rank matrix (rank × in_features)
//! - B is a low-rank matrix (out_features × rank)
//! - α is a scaling factor
use ndarray::{Array, ArrayView2, Axis, Ix2, s};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

/// Apply LoRA transformation to a tensor.
///
/// Implements the formula: W' = W + α(BA).
///
/// # Arguments
///
/// * `weight` - Original weight matrix (out_features × in_features)
/// * `lora_a` - Low-rank matrix A (rank × in_features)
/// * `lora_b` - Low-rank matrix B (out_features × rank)
/// * `alpha` - Scaling factor for the LoRA contribution
///
/// # Returns
///
/// The modified weight tensor with LoRA adaptation applied.
///
/// # Errors
///
/// Returns an error if matrix dimensions are incompatible.
///
/// # Examples
///
/// ```python
/// import lora_ops
/// import numpy as np
///
/// # Create sample weights and LoRA matrices
/// weight = np.random.randn(768, 768).astype(np.float32)
/// lora_a = np.random.randn(16, 768).astype(np.float32)
/// lora_b = np.random.randn(768, 16).astype(np.float32)
/// alpha = 1.0
///
/// # Apply LoRA adaptation
/// adapted_weight = lora_ops.apply_lora(weight, lora_a, lora_b, alpha)
/// ```
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

/// Compute the matrix product BA between two matrices.
///
/// # Arguments
///
/// * `b` - Matrix B (out_features × rank)
/// * `a` - Matrix A (rank × in_features)
///
/// # Returns
///
/// The product matrix BA of shape (out_features × in_features).
///
/// # Errors
///
/// Returns an error if matrix dimensions are incompatible for multiplication.
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

/// Apply multiple LoRA transformations to a tensor in a batch.
///
/// Processes multiple (A, B, α) triplets efficiently in a single operation.
///
/// # Arguments
///
/// * `weight` - Original weight matrix (out_features × in_features)
/// * `lora_pairs` - Vector of (A, B, α) triplets to apply sequentially
///
/// # Returns
///
/// The weight tensor with all LoRA adaptations applied.
///
/// # Errors
///
/// Returns an error if any matrix dimensions are incompatible.
///
/// # Examples
///
/// ```python
/// import lora_ops
/// import numpy as np
///
/// # Create sample weights and LoRA matrices
/// weight = np.random.randn(768, 768).astype(np.float32)
/// lora_pairs = [
///     (np.random.randn(16, 768).astype(np.float32), np.random.randn(768, 16).astype(np.float32), 1.0),
///     (np.random.randn(8, 768).astype(np.float32), np.random.randn(768, 8).astype(np.float32), 0.5)
/// ]
///
/// # Apply multiple LoRA adaptations
/// adapted_weight = lora_ops.apply_lora_batch(weight, lora_pairs)
/// ```
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

// ============= TRAINING FUNCTIONALITY =============

/// Adam optimizer parameters for LoRA training.
///
/// Manages the optimizer state including first and second moment estimates
/// for efficient adaptive learning rate adjustment.
///
/// # Fields
///
/// * `beta1` - Exponential decay rate for first moment estimates (default: 0.9)
/// * `beta2` - Exponential decay rate for second moment estimates (default: 0.999)
/// * `epsilon` - Small constant for numerical stability (default: 1e-8)
/// * `learning_rate` - Step size for parameter updates
/// * `m_a`, `v_a` - First and second moment estimates for matrix A
/// * `m_b`, `v_b` - First and second moment estimates for matrix B
/// * `step` - Number of update steps taken
#[pyclass]
struct AdamParams {
    #[pyo3(get, set)]
    beta1: f32,
    #[pyo3(get, set)]
    beta2: f32,
    #[pyo3(get, set)]
    epsilon: f32,
    #[pyo3(get, set)]
    learning_rate: f32,
    m_a: HashMap<String, Array<f32, Ix2>>,
    v_a: HashMap<String, Array<f32, Ix2>>,
    m_b: HashMap<String, Array<f32, Ix2>>,
    v_b: HashMap<String, Array<f32, Ix2>>,
    step: usize,
}

#[pymethods]
impl AdamParams {
    /// Create a new Adam optimizer with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for parameter updates
    /// * `beta1` - Exponential decay rate for first moment estimates
    /// * `beta2` - Exponential decay rate for second moment estimates
    /// * `epsilon` - Small constant for numerical stability
    ///
    /// # Returns
    ///
    /// A new AdamParams instance with empty moment estimates.
    #[new]
    fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            beta1,
            beta2,
            epsilon,
            learning_rate,
            m_a: HashMap::new(),
            v_a: HashMap::new(),
            m_b: HashMap::new(),
            v_b: HashMap::new(),
            step: 0,
        }
    }

    /// Reset the optimizer state.
    ///
    /// Clears all moment estimates and resets the step counter to zero.
    fn reset(&mut self) {
        self.m_a.clear();
        self.v_a.clear();
        self.m_b.clear();
        self.v_b.clear();
        self.step = 0;
    }
}

/// LoRA training context for a specific layer.
///
/// Manages the trainable LoRA parameters (A and B matrices) for a single layer,
/// handles forward and backward passes, and integrates with the optimizer.
///
/// # Fields
///
/// * `layer_name` - Unique identifier for the layer
/// * `rank` - Rank of the low-rank adaptation (r)
/// * `learning_rate` - Learning rate for SGD when optimizer is not used
/// * `alpha` - Scaling factor for LoRA contribution
/// * `lora_a` - Low-rank matrix A (rank × in_features)
/// * `lora_b` - Low-rank matrix B (out_features × rank)
/// * `frozen_weight` - Original weight matrix (fixed during training)
/// * `optimizer` - Optional Adam optimizer for parameter updates
#[pyclass]
struct LoraTrainingContext {
    #[pyo3(get, set)]
    layer_name: String,
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get, set)]
    learning_rate: f32,
    #[pyo3(get, set)]
    alpha: f32,
    lora_a: Array<f32, Ix2>,
    lora_b: Array<f32, Ix2>,
    frozen_weight: Array<f32, Ix2>,
    optimizer: Option<AdamParams>,
}

#[pymethods]
impl LoraTrainingContext {
    /// Create a new LoRA training context for a layer.
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Unique identifier for the layer
    /// * `weight` - Original weight matrix to adapt
    /// * `rank` - Rank of the low-rank adaptation (r)
    /// * `alpha` - Scaling factor for LoRA contribution
    /// * `init_scale` - Scale factor for random initialization of A
    ///
    /// # Returns
    ///
    /// A new LoraTrainingContext with initialized A and B matrices.
    ///
    /// # Examples
    ///
    /// ```python
    /// import lora_ops
    /// import numpy as np
    ///
    /// # Create weight matrix for a layer
    /// weight = np.random.randn(768, 768).astype(np.float32)
    ///
    /// # Create LoRA context
    /// ctx = lora_ops.LoraTrainingContext(
    ///     layer_name="transformer.h.0.attention.self.query",
    ///     weight=weight,
    ///     rank=16,
    ///     alpha=32,
    ///     init_scale=0.01
    /// )
    /// ```
    #[new]
    fn new<'py>(
        py: Python<'py>,
        layer_name: String,
        weight: PyReadonlyArray2<'py, f32>,
        rank: usize,
        alpha: f32,
        init_scale: f32,
    ) -> PyResult<Self> {
        let weight_view = weight.as_array();
        let weight_shape = weight_view.shape();
        let out_features = weight_shape[0];
        let in_features = weight_shape[1];
        
        println!("Initializing LoRA for layer {} with shape [{}, {}] and rank {}", 
                 layer_name, out_features, in_features, rank);
        
        // Initialize LoRA A with scaled random values (typically close to zero)
        let mut lora_a = Array::zeros((rank, in_features));
        for i in 0..rank {
            for j in 0..in_features {
                lora_a[[i, j]] = (rand::random::<f32>() - 0.5) * init_scale;
            }
        }
        
        // Initialize LoRA B with zeros (common initialization scheme)
        let lora_b = Array::zeros((out_features, rank));
        
        Ok(Self {
            layer_name,
            rank,
            learning_rate: 1e-3,
            alpha,
            lora_a,
            lora_b,
            frozen_weight: weight_view.to_owned(),
            optimizer: None,
        })
    }
    
    /// Set the optimizer for parameter updates.
    ///
    /// Initializes the optimizer state (moment estimates) for this layer's
    /// LoRA parameters.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Adam optimizer parameters
    ///
    /// # Returns
    ///
    /// PyResult indicating success or failure
    fn set_optimizer(&mut self, optimizer: &AdamParams) -> PyResult<()> {
        // Initialize optimizer states for this layer
        let mut opt = AdamParams::new(
            optimizer.learning_rate,
            optimizer.beta1,
            optimizer.beta2,
            optimizer.epsilon
        );
        
        // Initialize momentum and variance for A and B
        opt.m_a.insert(self.layer_name.clone(), Array::zeros(self.lora_a.dim()));
        opt.v_a.insert(self.layer_name.clone(), Array::zeros(self.lora_a.dim()));
        opt.m_b.insert(self.layer_name.clone(), Array::zeros(self.lora_b.dim()));
        opt.v_b.insert(self.layer_name.clone(), Array::zeros(self.lora_b.dim()));
        
        self.optimizer = Some(opt);
        Ok(())
    }
    
    /// Perform forward pass with LoRA adaptation.
    ///
    /// Applies the formula: W' = W + α(BA) to produce the adapted weights.
    ///
    /// # Returns
    ///
    /// The adapted weight matrix with LoRA transformation applied.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix dimensions are incompatible.
    fn forward<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray<f32, Ix2>> {
        let ba_product = compute_ba_product(&ArrayView2::from(&self.lora_b), 
                                           &ArrayView2::from(&self.lora_a))?;
        
        let mut result = self.frozen_weight.clone();
        
        // Add LoRA adjustment
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                result[[i, j]] += self.alpha * ba_product[[i, j]];
            }
        }
        
        Ok(result.into_pyarray(py))
    }
    
    /// Perform backward pass and update LoRA parameters.
    ///
    /// Computes gradients for A and B matrices based on grad_output,
    /// then applies updates using either Adam optimizer or simple SGD.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the loss with respect to the output (same shape as weight)
    ///
    /// # Returns
    ///
    /// PyResult indicating success or failure
    ///
    /// # Examples
    ///
    /// ```python
    /// # Assuming ctx is a LoraTrainingContext instance
    /// # and grad is the gradient from backpropagation
    /// ctx.backward(grad)
    /// ```
    fn backward<'py>(
        &mut self,
        py: Python<'py>,
        grad_output: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<()> {
        let grad_output_view = grad_output.as_array();
        
        // Calculate gradient for A: B^T * grad_output
        let b_t = self.lora_b.t();
        let grad_a = b_t.dot(&grad_output_view);
        
        // Calculate gradient for B: grad_output * A^T
        let a_t = self.lora_a.t();
        let grad_b = grad_output_view.dot(&a_t);
        
        // Apply gradients with optimizer if available
        if let Some(ref mut opt) = self.optimizer {
            opt.step += 1;
            
            // Update A parameters with Adam
            let m_a = opt.m_a.get_mut(&self.layer_name).unwrap();
            let v_a = opt.v_a.get_mut(&self.layer_name).unwrap();
            
            for i in 0..self.lora_a.shape()[0] {
                for j in 0..self.lora_a.shape()[1] {
                    // Update biased first moment estimate
                    m_a[[i, j]] = opt.beta1 * m_a[[i, j]] + (1.0 - opt.beta1) * grad_a[[i, j]];
                    
                    // Update biased second raw moment estimate
                    v_a[[i, j]] = opt.beta2 * v_a[[i, j]] + (1.0 - opt.beta2) * grad_a[[i, j]] * grad_a[[i, j]];
                    
                    // Bias correction
                    let m_hat = m_a[[i, j]] / (1.0 - opt.beta1.powi(opt.step as i32));
                    let v_hat = v_a[[i, j]] / (1.0 - opt.beta2.powi(opt.step as i32));
                    
                    // Update parameters
                    self.lora_a[[i, j]] -= opt.learning_rate * m_hat / (v_hat.sqrt() + opt.epsilon);
                }
            }
            
            // Update B parameters with Adam
            let m_b = opt.m_b.get_mut(&self.layer_name).unwrap();
            let v_b = opt.v_b.get_mut(&self.layer_name).unwrap();
            
            for i in 0..self.lora_b.shape()[0] {
                for j in 0..self.lora_b.shape()[1] {
                    // Update biased first moment estimate
                    m_b[[i, j]] = opt.beta1 * m_b[[i, j]] + (1.0 - opt.beta1) * grad_b[[i, j]];
                    
                    // Update biased second raw moment estimate
                    v_b[[i, j]] = opt.beta2 * v_b[[i, j]] + (1.0 - opt.beta2) * grad_b[[i, j]] * grad_b[[i, j]];
                    
                    // Bias correction
                    let m_hat = m_b[[i, j]] / (1.0 - opt.beta1.powi(opt.step as i32));
                    let v_hat = v_b[[i, j]] / (1.0 - opt.beta2.powi(opt.step as i32));
                    
                    // Update parameters
                    self.lora_b[[i, j]] -= opt.learning_rate * m_hat / (v_hat.sqrt() + opt.epsilon);
                }
            }
        } else {
            // Simple SGD without optimizer
            for i in 0..self.lora_a.shape()[0] {
                for j in 0..self.lora_a.shape()[1] {
                    self.lora_a[[i, j]] -= self.learning_rate * grad_a[[i, j]];
                }
            }
            
            for i in 0..self.lora_b.shape()[0] {
                for j in 0..self.lora_b.shape()[1] {
                    self.lora_b[[i, j]] -= self.learning_rate * grad_b[[i, j]];
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the current LoRA weight matrices.
    ///
    /// Returns the current values of the A and B matrices for saving
    /// or transfer to other models.
    ///
    /// # Returns
    ///
    /// A tuple of (A, B) matrices as NumPy arrays.
    ///
    /// # Examples
    ///
    /// ```python
    /// # Save LoRA weights to file
    /// lora_a, lora_b = ctx.get_weights()
    /// np.save("lora_a.npy", lora_a)
    /// np.save("lora_b.npy", lora_b)
    /// ```
    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<(
        &'py PyArray<f32, Ix2>,
        &'py PyArray<f32, Ix2>
    )> {
        Ok((
            self.lora_a.to_owned().into_pyarray(py),
            self.lora_b.to_owned().into_pyarray(py)
        ))
    }
}

/// Register module functions and classes with Python.
#[pymodule]
fn lora_ops(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_lora, m)?)?;
    m.add_function(wrap_pyfunction!(apply_lora_batch, m)?)?;
    m.add_class::<LoraTrainingContext>()?;
    m.add_class::<AdamParams>()?;
    Ok(())
} 