import { useState, useEffect, useRef } from 'react'
import * as d3 from 'd3'

const GradientFlow = () => {
  const svgRef = useRef<SVGSVGElement>(null)
  const [isAnimating, setIsAnimating] = useState(false)
  const [step, setStep] = useState(0)
  
  // Steps in the backpropagation process
  const totalSteps = 4
  
  const startAnimation = () => {
    setIsAnimating(true)
    setStep(0)
  }
  
  const stopAnimation = () => {
    setIsAnimating(false)
  }
  
  const nextStep = () => {
    setStep(prev => (prev + 1) % (totalSteps + 1))
  }
  
  const prevStep = () => {
    setStep(prev => (prev - 1 + (totalSteps + 1)) % (totalSteps + 1))
  }
  
  useEffect(() => {
    if (isAnimating) {
      const timer = setTimeout(() => {
        setStep(prev => (prev + 1) % (totalSteps + 1))
      }, 1500)
      
      return () => clearTimeout(timer)
    }
  }, [step, isAnimating])
  
  useEffect(() => {
    if (!svgRef.current) return
    
    // Clear previous svg content
    d3.select(svgRef.current).selectAll('*').remove()
    
    // SVG dimensions
    const width = 800
    const height = 500
    const margin = { top: 40, right: 20, bottom: 40, left: 20 }
    
    // Create the SVG container
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .attr('style', 'max-width: 100%; height: auto;')
    
    // Define colors
    const colors = {
      frozen: '#d4f1f9',
      trainable: '#ffcccc',
      active: '#e74c3c',
      inactive: '#95a5a6',
      gradient: '#9b59b6',
      loss: '#f39c12'
    }
    
    // Function to create a node box
    const createNode = (x, y, width, height, label, color, isActive = true) => {
      const g = svg.append('g')
      
      // Add box
      g.append('rect')
        .attr('x', x)
        .attr('y', y)
        .attr('width', width)
        .attr('height', height)
        .attr('rx', 5)
        .attr('ry', 5)
        .attr('fill', color)
        .attr('stroke', isActive ? '#2c3e50' : '#bdc3c7')
        .attr('stroke-width', 2)
        .attr('opacity', isActive ? 1 : 0.7)
      
      // Add label
      g.append('text')
        .attr('x', x + width / 2)
        .attr('y', y + height / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .attr('fill', '#2c3e50')
        .attr('opacity', isActive ? 1 : 0.7)
        .text(label)
      
      return g
    }
    
    // Function to create an arrow
    const createArrow = (x1, y1, x2, y2, color = '#2c3e50', isActive = true, isGradient = false) => {
      const g = svg.append('g')
      
      // Create arrow line
      const line = g.append('line')
        .attr('x1', x1)
        .attr('y1', y1)
        .attr('x2', x2)
        .attr('y2', y2)
        .attr('stroke', color)
        .attr('stroke-width', isGradient ? 3 : 2)
        .attr('opacity', isActive ? 1 : 0.4)
      
      // Add arrowhead
      if (isGradient) {
        line.attr('marker-end', 'url(#gradient-arrow)')
        line.attr('stroke-dasharray', '5,3')
      } else {
        line.attr('marker-end', 'url(#arrow)')
      }
      
      return g
    }
    
    // Define arrow markers
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#2c3e50')
    
    // Gradient arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'gradient-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', colors.gradient)
    
    // Create the neural network architecture
    // Loss node
    createNode(width / 2 - 60, 50, 120, 40, 'Loss Function', colors.loss)
    
    // UNet output node
    createNode(width / 2 - 60, 140, 120, 40, 'UNet Output', colors.inactive, step >= 1)
    
    // Base model weights node (left branch)
    createNode(width / 4 - 70, 230, 140, 40, 'Original Weights', colors.frozen, step >= 2)
    
    // LoRA matrices node (right branch)
    createNode(3 * width / 4 - 70, 230, 140, 40, 'LoRA Matrices', colors.trainable, step >= 2)
    
    // Frozen weights output node
    createNode(width / 4 - 60, 320, 120, 40, 'No Update', colors.frozen, step >= 3)
    
    // LoRA update node
    createNode(3 * width / 4 - 60, 320, 120, 40, 'Update Weights', colors.active, step >= 3)
    
    // Forward pass arrows
    createArrow(width / 2, 90, width / 2, 140, colors.inactive)
    
    // Backprop arrows - only show based on current step
    const lossToOutput = createArrow(width / 2, 90, width / 2, 140, colors.gradient, step >= 1, true)
    lossToOutput.attr('opacity', step >= 1 ? 1 : 0)
    
    // Backprop to branches
    const outputToFrozen = createArrow(width / 2 - 20, 180, width / 4, 230, colors.gradient, step >= 2, true)
    outputToFrozen.attr('opacity', step >= 2 ? 1 : 0)
    
    const outputToLoRA = createArrow(width / 2 + 20, 180, 3 * width / 4, 230, colors.gradient, step >= 2, true)
    outputToLoRA.attr('opacity', step >= 2 ? 1 : 0)
    
    // Final update arrows
    const frozenToNoUpdate = createArrow(width / 4, 270, width / 4, 320, colors.inactive, step >= 3)
    frozenToNoUpdate.attr('opacity', step >= 3 ? 1 : 0)
    
    const loraToUpdate = createArrow(3 * width / 4, 270, 3 * width / 4, 320, colors.active, step >= 3)
    loraToUpdate.attr('opacity', step >= 3 ? 1 : 0)
    
    // Add step description
    const stepTexts = [
      "Initial state: LoRA training starts with calculating loss.",
      "Step 1: Gradients flow backward from the loss function to the UNet output.",
      "Step 2: Gradients split towards original weights and LoRA matrices.",
      "Step 3: Only LoRA matrices are updated, original weights remain frozen.",
      "Complete: This parameter-efficient approach is the key to LoRA's effectiveness."
    ]
    
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text(stepTexts[step])
    
    // Formula for gradient calculation
    if (step >= 2) {
      const formulaG = svg.append('g')
        .attr('opacity', step >= 2 ? 1 : 0)
      
      formulaG.append('text')
        .attr('x', width / 2)
        .attr('y', 390)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('font-style', 'italic')
        .text("Gradient calculations:")
      
      formulaG.append('text')
        .attr('x', width / 4)
        .attr('y', 420)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-style', 'italic')
        .text("∂L/∂W (calculated but ignored)")
      
      formulaG.append('text')
        .attr('x', 3 * width / 4)
        .attr('y', 420)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-style', 'italic')
        .text("∂L/∂B, ∂L/∂A (applied to LoRA matrices)")
    }
    
    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text('Gradient Flow in LoRA Training')
    
  }, [step])
  
  return (
    <div className="diagram-container">
      <h2>Gradient Flow Visualization</h2>
      <p>
        This interactive visualization shows how gradients flow during backpropagation in LoRA training.
        The key innovation in LoRA is that gradients only update the small adapter matrices (B and A) while
        the original weights remain frozen.
      </p>
      
      <div className="interactive-controls">
        {isAnimating ? (
          <button className="button" onClick={stopAnimation}>Stop Animation</button>
        ) : (
          <button className="button" onClick={startAnimation}>Start Animation</button>
        )}
        
        <button className="button" onClick={prevStep} disabled={isAnimating}>Previous Step</button>
        <button className="button" onClick={nextStep} disabled={isAnimating}>Next Step</button>
        
        <div className="step-indicator">
          Step: {step} / {totalSteps}
        </div>
      </div>
      
      <svg ref={svgRef}></svg>
      
      <div className="key-formulas">
        <h3>Key LoRA Gradient Formulas</h3>
        <p>
          For LoRA matrices A and B in the equation W' = W + BA × (α/r):
        </p>
        <ul>
          <li><strong>∂L/∂B</strong> = ∂L/∂y × (A×x)ᵀ × (α/r)</li>
          <li><strong>∂L/∂A</strong> = Bᵀ × ∂L/∂y × xᵀ × (α/r)</li>
        </ul>
        <p>
          These gradients are used to update only the LoRA matrices, while the original weights W remain unchanged.
        </p>
      </div>
    </div>
  )
}

export default GradientFlow 