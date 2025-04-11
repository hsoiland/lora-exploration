import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

const FullProcessDiagram = () => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    // Clear previous svg content
    d3.select(svgRef.current).selectAll('*').remove()

    // SVG dimensions
    const width = 800
    const height = 700
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
      latent: '#e6f7ff',
      noise: '#ffebcc',
      model: '#e6ffe6',
      loss: '#ffe6e6',
      prompt: '#fff8e1',
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
      .text('Complete SDXL Training Process with LoRA')

    // Function to create a component box
    const createComponent = (x, y, width, height, title, detail = '', color = '#fff', borderColor = '#333') => {
      const group = g.append('g')
        .attr('transform', `translate(${x},${y})`)

      // Draw box
      group.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('rx', 5)
        .attr('ry', 5)
        .attr('fill', color)
        .attr('stroke', borderColor)
        .attr('stroke-width', 2)

      // Add title
      group.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .attr('fill', '#2c3e50')
        .text(title)

      // Add detail if provided
      if (detail) {
        group.append('text')
          .attr('x', width / 2)
          .attr('y', height / 2 + 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', '12px')
          .attr('fill', '#2c3e50')
          .text(detail)
      }

      return group
    }

    // Create connection arrow
    const createArrow = (x1, y1, x2, y2, color = colors.arrow, dashed = false) => {
      const path = g.append('path')
        .attr('d', `M${x1},${y1} L${x2},${y2}`)
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrow)')

      if (dashed) {
        path.attr('stroke-dasharray', '5,5')
      }

      return path
    }

    // Define arrow marker
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

    // Layout constants
    const col1X = 50
    const col2X = 300
    const col3X = 550
    const boxWidth = 200
    const boxHeight = 60
    const rowSpacing = 90

    // Row 1: Input data
    const imageRow = 80
    const imageBox = createComponent(col1X, imageRow, boxWidth, boxHeight, 'Input Image', '', colors.prompt, '#d35400')
    const captionBox = createComponent(col3X, imageRow, boxWidth, boxHeight, 'Input Caption', '', colors.prompt, '#d35400')

    // Row 2: Encoders
    const encoderRow = imageRow + rowSpacing
    const vaeEncoderBox = createComponent(col1X, encoderRow, boxWidth, boxHeight, 'VAE Encoder', '[FROZEN]', colors.frozen, '#7f8c8d')
    const textEncoderBox = createComponent(col3X, encoderRow, boxWidth, boxHeight, 'Text Encoders', '[FROZEN]', colors.frozen, '#7f8c8d')

    // Row 3: Encoded forms
    const encodedRow = encoderRow + rowSpacing
    const latentBox = createComponent(col1X, encodedRow, boxWidth, boxHeight, 'Latent Representation', 'z₀', colors.latent, '#3498db')
    const embeddingBox = createComponent(col3X, encodedRow, boxWidth, boxHeight, 'Text Embeddings', '', colors.prompt, '#d35400')

    // Row 4: Add noise & timestep
    const noiseRow = encodedRow + rowSpacing
    const noiseBox = createComponent(col1X, noiseRow, boxWidth, boxHeight, 'Add Noise', 'timestep t', colors.noise, '#f39c12')
    createArrow(col1X + boxWidth / 2, encodedRow + boxHeight, col1X + boxWidth / 2, noiseRow)

    // Row 5: UNet with LoRA
    const unetRow = noiseRow + rowSpacing
    const unetBox = createComponent(col2X, unetRow, boxWidth, boxHeight, 'UNet with LoRA', '', colors.model, '#2ecc71')
    
    // Draw UNet detail
    const unetDetailX = col2X + boxWidth + 20
    const unetDetailY = unetRow - 30
    const unetDetailWidth = 180
    const unetDetailHeight = 120
    
    g.append('rect')
      .attr('x', unetDetailX)
      .attr('y', unetDetailY)
      .attr('width', unetDetailWidth)
      .attr('height', unetDetailHeight)
      .attr('rx', 5)
      .attr('fill', '#f8f9fa')
      .attr('stroke', '#3498db')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
    
    g.append('text')
      .attr('x', unetDetailX + 10)
      .attr('y', unetDetailY + 20)
      .attr('fill', '#2c3e50')
      .attr('font-weight', 'bold')
      .text('UNet Architecture:')
    
    g.append('text')
      .attr('x', unetDetailX + 15)
      .attr('y', unetDetailY + 45)
      .attr('fill', '#7f8c8d')
      .attr('font-size', '12px')
      .text('• Base weights: [FROZEN]')
    
    g.append('text')
      .attr('x', unetDetailX + 15)
      .attr('y', unetDetailY + 65)
      .attr('fill', '#e74c3c')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text('• LoRA adapters: [TRAINABLE]')
    
    g.append('text')
      .attr('x', unetDetailX + 15)
      .attr('y', unetDetailY + 85)
      .attr('fill', '#2c3e50')
      .attr('font-size', '12px')
      .text('• Rank: 16')
    
    g.append('text')
      .attr('x', unetDetailX + 15)
      .attr('y', unetDetailY + 105)
      .attr('fill', '#2c3e50')
      .attr('font-size', '12px')
      .text('• Target: Attention layers only')
    
    // Connect to UNet
    createArrow(unetDetailX, unetDetailY + unetDetailHeight / 2, unetDetailX - 20, unetRow + boxHeight / 2)

    // Connect inputs to UNet
    createArrow(col1X + boxWidth, noiseRow + boxHeight / 2, col2X, unetRow + boxHeight / 3)
    createArrow(col3X + boxWidth / 2, encodedRow + boxHeight, col2X + boxWidth / 2, unetRow - 10)
    
    // Row 6: Noise prediction
    const predictionRow = unetRow + rowSpacing
    const predictedNoiseBox = createComponent(col1X, predictionRow, boxWidth, boxHeight, 'Predicted Noise', 'ε_θ', colors.model, '#2ecc71')
    const actualNoiseBox = createComponent(col3X, predictionRow, boxWidth, boxHeight, 'Actual Noise', 'ε', colors.noise, '#f39c12')
    
    // Connect UNet to prediction
    createArrow(col2X + boxWidth / 2, unetRow + boxHeight, col1X + boxWidth / 2, predictionRow)
    
    // Connect noise to actual noise (store the original noise)
    const noiseSaver = createArrow(col1X + boxWidth, noiseRow + boxHeight / 2, col3X, predictionRow - 10)
    noiseSaver.attr('stroke-dasharray', '5,5')

    // Row 7: Loss calculation
    const lossRow = predictionRow + rowSpacing
    const lossBox = createComponent(col2X, lossRow, boxWidth, boxHeight, 'MSE Loss', 'Compare prediction with target', colors.loss, '#e74c3c')
    
    // Connect to loss
    createArrow(col1X + boxWidth, predictionRow + boxHeight / 2, col2X, lossRow + boxHeight / 2)
    createArrow(col3X, predictionRow + boxHeight / 2, col2X + boxWidth, lossRow + boxHeight / 2)
    
    // Row 8: Backpropagation
    const backpropRow = lossRow + rowSpacing
    const backpropBox = createComponent(col2X, backpropRow, boxWidth, boxHeight, 'Backpropagation', 'Update only LoRA parameters', colors.trainable, '#e74c3c')
    
    // Connect loss to backprop
    createArrow(col2X + boxWidth / 2, lossRow + boxHeight, col2X + boxWidth / 2, backpropRow)
    
    // Add legend for update path
    g.append('rect')
      .attr('x', 50)
      .attr('y', backpropRow + boxHeight + 20)
      .attr('width', width - 100)
      .attr('height', 50)
      .attr('rx', 5)
      .attr('fill', '#f8f9fa')
      .attr('stroke', '#3498db')
      .attr('stroke-width', 2)
    
    g.append('text')
      .attr('x', width / 2)
      .attr('y', backpropRow + boxHeight + 40)
      .attr('text-anchor', 'middle')
      .attr('font-weight', 'bold')
      .attr('fill', '#2c3e50')
      .text('Only LoRA matrices A and B are updated (~1% of all parameters)')
    
    g.append('text')
      .attr('x', width / 2)
      .attr('y', backpropRow + boxHeight + 60)
      .attr('text-anchor', 'middle')
      .attr('font-style', 'italic')
      .attr('fill', '#2c3e50')
      .text('W\' = W + BA × (α/r), where only B and A are trainable')

  }, [])

  return (
    <div className="diagram-container">
      <h2>Complete SDXL Training Process with LoRA</h2>
      <p>
        This diagram provides an end-to-end view of the SDXL training process with LoRA adapters.
        It shows how all components work together, from the initial image and caption inputs
        to the final parameter updates.
      </p>
      <svg ref={svgRef}></svg>
      <div className="key-points">
        <h3>Training Process Summary</h3>
        <ol>
          <li><strong>Data Preparation</strong>: Images and captions from your datasets</li>
          <li><strong>Encoding</strong>: VAE encodes images to latents, text encoders process captions</li>
          <li><strong>Noise Addition</strong>: Random noise added to latents according to a timestep</li>
          <li><strong>UNet Forward Pass</strong>: UNet with LoRA adapters predicts the noise</li>
          <li><strong>Loss Calculation</strong>: MSE between predicted and original noise</li>
          <li><strong>Backpropagation</strong>: Gradients flow back, updating only LoRA parameters</li>
        </ol>
        <p>
          By training only the small LoRA matrices (~1% of all parameters), you can efficiently
          fine-tune SDXL on your custom datasets (Gina and Ilya Repin) while preserving the
          base model's capabilities.
        </p>
      </div>
    </div>
  )
}

export default FullProcessDiagram 