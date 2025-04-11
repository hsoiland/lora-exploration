import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

const BackpropDiagram = () => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    // Clear previous svg content
    d3.select(svgRef.current).selectAll('*').remove()

    // SVG dimensions
    const width = 800
    const height = 600
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }

    // Create the SVG container
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .attr('style', 'max-width: 100%; height: auto;')

    // Colors
    const colors = {
      frozen: '#d4f1f9',
      trainable: '#ffcccc',
      gradient: '#9b59b6',
      loss: '#f39c12',
      output: '#e8f5e9',
      arrow: '#95a5a6',
      highlight: '#3498db'
    }

    // Main container group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '20px')
      .attr('font-weight', 'bold')
      .text('Backpropagation Through LoRA')

    // Draw the gradient flow diagram
    // This will be a tree-like structure showing how gradients flow during backprop

    // Draw nodes
    const boxWidth = 160
    const boxHeight = 50
    const centerX = width / 2
    const startY = 80
    const verticalSpace = 80

    // Function to create a box
    const createBox = (x, y, width, height, text, color, borderColor = 'black') => {
      const group = g.append('g')
        .attr('transform', `translate(${x},${y})`)

      group.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('rx', 5)
        .attr('ry', 5)
        .attr('fill', color)
        .attr('stroke', borderColor)
        .attr('stroke-width', 2)

      group.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-weight', 'bold')
        .text(text)

      return group
    }

    // Create the main flow
    const lossNode = createBox(centerX - boxWidth / 2, startY, boxWidth, boxHeight, 'Loss Function', colors.loss, '#d35400')
    const outputNode = createBox(centerX - boxWidth / 2, startY + verticalSpace, boxWidth, boxHeight, 'UNet Output', colors.output, colors.highlight)
    
    // Create the branching nodes
    const leftX = centerX - boxWidth - 40
    const rightX = centerX + 40
    const branchY = startY + verticalSpace * 2
    
    const originalNode = createBox(leftX, branchY, boxWidth, boxHeight, 'Original Weights', colors.frozen, '#95a5a6')
    const loraNode = createBox(rightX, branchY, boxWidth, boxHeight, 'LoRA Matrices', colors.trainable, '#e74c3c')
    
    // Create the outcome nodes
    const noUpdateNode = createBox(leftX, branchY + verticalSpace, boxWidth, boxHeight, 'No Update', colors.frozen, '#95a5a6')
    const updateNode = createBox(rightX, branchY + verticalSpace, boxWidth, boxHeight, 'Parameter Update', '#e74c3c', '#c0392b')
    
    // Add arrows for forward pass (light)
    g.append('path')
      .attr('d', `M${centerX},${startY + boxHeight} L${centerX},${startY + verticalSpace}`)
      .attr('stroke', colors.arrow)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,4')
      .attr('marker-end', 'url(#arrow)')
    
    // Add arrows for backward pass (gradients)
    // Loss to Output
    g.append('path')
      .attr('d', `M${centerX},${startY + boxHeight} L${centerX},${startY + verticalSpace}`)
      .attr('stroke', colors.gradient)
      .attr('stroke-width', 3)
      .attr('marker-end', 'url(#gradient-arrow)')
    
    // Output to branches
    g.append('path')
      .attr('d', `M${centerX - 20},${startY + verticalSpace + boxHeight} L${leftX + boxWidth / 2},${branchY}`)
      .attr('stroke', colors.gradient)
      .attr('stroke-width', 3)
      .attr('marker-end', 'url(#gradient-arrow)')
    
    g.append('path')
      .attr('d', `M${centerX + 20},${startY + verticalSpace + boxHeight} L${rightX + boxWidth / 2},${branchY}`)
      .attr('stroke', colors.gradient)
      .attr('stroke-width', 3)
      .attr('marker-end', 'url(#gradient-arrow)')
    
    // Branches to outcomes
    g.append('path')
      .attr('d', `M${leftX + boxWidth / 2},${branchY + boxHeight} L${leftX + boxWidth / 2},${branchY + verticalSpace}`)
      .attr('stroke', '#bdc3c7')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('marker-end', 'url(#blocked-arrow)')
    
    g.append('path')
      .attr('d', `M${rightX + boxWidth / 2},${branchY + boxHeight} L${rightX + boxWidth / 2},${branchY + verticalSpace}`)
      .attr('stroke', colors.highlight)
      .attr('stroke-width', 3)
      .attr('marker-end', 'url(#update-arrow)')
    
    // Add gradient operators
    g.append('text')
      .attr('x', centerX - 50)
      .attr('y', startY + verticalSpace - 15)
      .attr('font-size', '16px')
      .attr('font-style', 'italic')
      .attr('fill', colors.gradient)
      .text('∂L/∂y')
    
    g.append('text')
      .attr('x', leftX + boxWidth / 2 - 60)
      .attr('y', branchY - 15)
      .attr('font-size', '16px')
      .attr('font-style', 'italic')
      .attr('fill', colors.gradient)
      .text('∂L/∂W')
    
    g.append('text')
      .attr('x', rightX + boxWidth / 2 - 60)
      .attr('y', branchY - 15)
      .attr('font-size', '16px')
      .attr('font-style', 'italic')
      .attr('fill', colors.gradient)
      .text('∂L/∂(BA)')
    
    // Add text descriptions
    g.append('text')
      .attr('x', leftX + boxWidth / 2)
      .attr('y', branchY + verticalSpace + boxHeight + 15)
      .attr('text-anchor', 'middle')
      .attr('font-weight', 'bold')
      .attr('fill', '#95a5a6')
      .text('Gradients calculated but not applied')
    
    g.append('text')
      .attr('x', rightX + boxWidth / 2)
      .attr('y', branchY + verticalSpace + boxHeight + 15)
      .attr('text-anchor', 'middle')
      .attr('font-weight', 'bold')
      .attr('fill', '#e74c3c')
      .text('Gradients applied to update parameters')
    
    // Add formula section
    const formulaY = branchY + verticalSpace + 90
    g.append('rect')
      .attr('x', width / 2 - 300)
      .attr('y', formulaY)
      .attr('width', 600)
      .attr('height', 80)
      .attr('rx', 5)
      .attr('fill', '#f8f9fa')
      .attr('stroke', colors.highlight)
      .attr('stroke-width', 2)
    
    g.append('text')
      .attr('x', width / 2)
      .attr('y', formulaY + 20)
      .attr('text-anchor', 'middle')
      .attr('font-weight', 'bold')
      .text('Gradient Calculation for LoRA Matrices')
    
    g.append('text')
      .attr('x', width / 2)
      .attr('y', formulaY + 45)
      .attr('text-anchor', 'middle')
      .attr('font-style', 'italic')
      .text('∂L/∂B = ∂L/∂y × (A×x)ᵀ × (α/r)')
    
    g.append('text')
      .attr('x', width / 2)
      .attr('y', formulaY + 65)
      .attr('text-anchor', 'middle')
      .attr('font-style', 'italic')
      .text('∂L/∂A = Bᵀ × ∂L/∂y × xᵀ × (α/r)')
    
    // Add arrowheads
    // Regular arrow
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', colors.arrow)
    
    // Gradient arrow
    svg.append('defs').append('marker')
      .attr('id', 'gradient-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', colors.gradient)
    
    // Blocked arrow
    svg.append('defs').append('marker')
      .attr('id', 'blocked-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#bdc3c7')
    
    // Update arrow
    svg.append('defs').append('marker')
      .attr('id', 'update-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', colors.highlight)

  }, [])

  return (
    <div className="diagram-container">
      <h2>Backpropagation Through LoRA</h2>
      <p>
        One of the key innovations in LoRA is how it handles backpropagation. During training,
        gradients flow throughout the network, but parameter updates are selectively applied
        only to the LoRA matrices, while the original weights remain frozen.
      </p>
      <svg ref={svgRef}></svg>
      <div className="key-points">
        <h3>Key Backpropagation Concepts</h3>
        <ul>
          <li><strong>Selective parameter updates</strong>: Gradients are calculated for all parameters but only applied to LoRA matrices</li>
          <li><strong>Parameter efficiency</strong>: By only updating a small fraction of parameters, we get massive efficiency gains</li>
          <li><strong>Gradient flow</strong>: Loss gradient flows back from the output layer to all trainable parameters</li>
          <li><strong>Gradient calculation</strong>: Special formulas account for the low-rank structure of LoRA adapters</li>
          <li><strong>Preservation of knowledge</strong>: By keeping the base model frozen, we retain all its general capabilities</li>
        </ul>
      </div>
      <div className="resource-links">
        <h3>Mathematical Details</h3>
        <p>
          For those interested in the mathematical foundations of LoRA, the key gradient calculations are:
        </p>
        <pre>∂L/∂B = ∂L/∂y × (A×x)ᵀ × (α/r)</pre>
        <pre>∂L/∂A = Bᵀ × ∂L/∂y × xᵀ × (α/r)</pre>
        <p>
          These gradients are used in the standard weight update rule: parameter = parameter - learning_rate × gradient
        </p>
      </div>
    </div>
  )
}

export default BackpropDiagram 