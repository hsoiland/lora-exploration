import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

const UNetDiagram = () => {
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
      input: '#fff8e1',
      output: '#e8f5e9',
      arrow: '#95a5a6',
      highlight: '#3498db'
    }

    // Main container group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Draw UNet architecture with LoRA

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '20px')
      .attr('font-weight', 'bold')
      .text('UNet Forward Pass with LoRA')

    // Draw LoRA in a single attention layer - left side
    const leftX = 120
    const topY = 100
    const boxWidth = 140
    const boxHeight = 40
    const spacing = 60

    // Draw input node
    g.append('rect')
      .attr('x', leftX)
      .attr('y', topY)
      .attr('width', boxWidth)
      .attr('height', boxHeight)
      .attr('rx', 5)
      .attr('fill', colors.input)
      .attr('stroke', colors.highlight)
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', leftX + boxWidth / 2)
      .attr('y', topY + boxHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-weight', 'bold')
      .text('Input Features')

    // Draw original weights - frozen path
    g.append('rect')
      .attr('x', leftX)
      .attr('y', topY + spacing)
      .attr('width', boxWidth)
      .attr('height', boxHeight)
      .attr('rx', 5)
      .attr('fill', colors.frozen)
      .attr('stroke', '#95a5a6')
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', leftX + boxWidth / 2)
      .attr('y', topY + spacing + boxHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-weight', 'bold')
      .text('Original Weights')

    g.append('text')
      .attr('x', leftX + boxWidth / 2)
      .attr('y', topY + spacing + boxHeight - 5)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'bottom')
      .attr('font-size', '10px')
      .attr('font-style', 'italic')
      .text('[FROZEN]')

    // Draw original output
    g.append('rect')
      .attr('x', leftX)
      .attr('y', topY + spacing * 2)
      .attr('width', boxWidth)
      .attr('height', boxHeight)
      .attr('rx', 5)
      .attr('fill', colors.frozen)
      .attr('stroke', '#95a5a6')
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', leftX + boxWidth / 2)
      .attr('y', topY + spacing * 2 + boxHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-weight', 'bold')
      .text('Original Output')

    // Draw LoRA path - right side
    const rightX = leftX + boxWidth + 80

    // Draw Matrix A
    g.append('rect')
      .attr('x', rightX)
      .attr('y', topY + spacing)
      .attr('width', boxWidth)
      .attr('height', boxHeight)
      .attr('rx', 5)
      .attr('fill', colors.trainable)
      .attr('stroke', '#e74c3c')
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', rightX + boxWidth / 2)
      .attr('y', topY + spacing + boxHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-weight', 'bold')
      .text('Matrix A')

    g.append('text')
      .attr('x', rightX + boxWidth / 2)
      .attr('y', topY + spacing + boxHeight - 5)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'bottom')
      .attr('font-size', '10px')
      .attr('font-style', 'italic')
      .text('[Rank × Input_dim]')

    // Draw Matrix B
    g.append('rect')
      .attr('x', rightX)
      .attr('y', topY + spacing * 1.5)
      .attr('width', boxWidth)
      .attr('height', boxHeight)
      .attr('rx', 5)
      .attr('fill', colors.trainable)
      .attr('stroke', '#e74c3c')
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', rightX + boxWidth / 2)
      .attr('y', topY + spacing * 1.5 + boxHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-weight', 'bold')
      .text('Matrix B')

    g.append('text')
      .attr('x', rightX + boxWidth / 2)
      .attr('y', topY + spacing * 1.5 + boxHeight - 5)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'bottom')
      .attr('font-size', '10px')
      .attr('font-style', 'italic')
      .text('[Output_dim × Rank]')

    // Draw scaled output
    g.append('rect')
      .attr('x', rightX)
      .attr('y', topY + spacing * 2)
      .attr('width', boxWidth)
      .attr('height', boxHeight)
      .attr('rx', 5)
      .attr('fill', colors.trainable)
      .attr('stroke', '#e74c3c')
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', rightX + boxWidth / 2)
      .attr('y', topY + spacing * 2 + boxHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-weight', 'bold')
      .text('LoRA Output × (α/r)')

    // Draw combined output
    const combinedX = (leftX + rightX + boxWidth) / 2 - boxWidth / 2
    const combinedY = topY + spacing * 3

    g.append('rect')
      .attr('x', combinedX)
      .attr('y', combinedY)
      .attr('width', boxWidth)
      .attr('height', boxHeight)
      .attr('rx', 5)
      .attr('fill', colors.output)
      .attr('stroke', colors.highlight)
      .attr('stroke-width', 2)

    g.append('text')
      .attr('x', combinedX + boxWidth / 2)
      .attr('y', combinedY + boxHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-weight', 'bold')
      .text('Combined Output')

    // Draw arrows
    // Input to paths
    g.append('path')
      .attr('d', `M${leftX + boxWidth / 2},${topY + boxHeight} L${leftX + boxWidth / 2},${topY + spacing}`)
      .attr('stroke', colors.arrow)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrow)')

    g.append('path')
      .attr('d', `M${leftX + boxWidth / 2},${topY + boxHeight} Q${leftX + boxWidth + 40},${topY + boxHeight + 20} ${rightX + boxWidth / 2},${topY + spacing}`)
      .attr('stroke', colors.arrow)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrow)')

    // Original weights to output
    g.append('path')
      .attr('d', `M${leftX + boxWidth / 2},${topY + spacing + boxHeight} L${leftX + boxWidth / 2},${topY + spacing * 2}`)
      .attr('stroke', colors.arrow)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrow)')

    // Matrix A to B
    g.append('path')
      .attr('d', `M${rightX + boxWidth / 2},${topY + spacing + boxHeight} L${rightX + boxWidth / 2},${topY + spacing * 1.5}`)
      .attr('stroke', colors.arrow)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrow)')

    // Matrix B to Scaled output
    g.append('path')
      .attr('d', `M${rightX + boxWidth / 2},${topY + spacing * 1.5 + boxHeight} L${rightX + boxWidth / 2},${topY + spacing * 2}`)
      .attr('stroke', colors.arrow)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrow)')

    // Outputs to Combined
    g.append('path')
      .attr('d', `M${leftX + boxWidth / 2},${topY + spacing * 2 + boxHeight} L${leftX + boxWidth / 2},${combinedY + boxHeight / 2} L${combinedX},${combinedY + boxHeight / 2}`)
      .attr('stroke', colors.arrow)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrow)')

    g.append('path')
      .attr('d', `M${rightX + boxWidth / 2},${topY + spacing * 2 + boxHeight} L${rightX + boxWidth / 2},${combinedY + boxHeight / 2} L${combinedX + boxWidth},${combinedY + boxHeight / 2}`)
      .attr('stroke', colors.arrow)
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .attr('marker-end', 'url(#arrow)')

    // Draw plus sign for addition
    g.append('text')
      .attr('x', combinedX + boxWidth / 2)
      .attr('y', combinedY + boxHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '25px')
      .attr('font-weight', 'bold')
      .text('+')

    // Add formula text
    g.append('text')
      .attr('x', width / 2)
      .attr('y', combinedY + boxHeight + 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-style', 'italic')
      .text('W\' = W + BA × (α/r)')

    // Add arrowheads for all paths
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

    // Target modules section - right side
    const moduleX = 500
    const moduleY = 100
    const moduleWidth = 220
    const moduleHeight = 30
    const moduleSpacing = 40

    g.append('text')
      .attr('x', moduleX + moduleWidth / 2)
      .attr('y', moduleY - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('LoRA Target Modules in SDXL')

    // Module list with colors indicating which ones get LoRA
    const modules = [
      { name: 'to_q (Query projection)', lora: true },
      { name: 'to_k (Key projection)', lora: true },
      { name: 'to_v (Value projection)', lora: true },
      { name: 'to_out.0 (Output projection)', lora: true },
      { name: 'ff.net (Feed-forward)', lora: false },
      { name: 'conv (Convolutional)', lora: false }
    ]

    modules.forEach((module, i) => {
      // Draw module box
      g.append('rect')
        .attr('x', moduleX)
        .attr('y', moduleY + i * moduleSpacing)
        .attr('width', moduleWidth)
        .attr('height', moduleHeight)
        .attr('rx', 5)
        .attr('fill', module.lora ? colors.trainable : colors.frozen)
        .attr('stroke', module.lora ? '#e74c3c' : '#95a5a6')
        .attr('stroke-width', 2)

      // Add module text
      g.append('text')
        .attr('x', moduleX + 10)
        .attr('y', moduleY + i * moduleSpacing + moduleHeight / 2)
        .attr('dominant-baseline', 'middle')
        .attr('font-weight', module.lora ? 'bold' : 'normal')
        .text(module.name)

      // Add LoRA indicator
      if (module.lora) {
        g.append('text')
          .attr('x', moduleX + moduleWidth - 10)
          .attr('y', moduleY + i * moduleSpacing + moduleHeight / 2)
          .attr('text-anchor', 'end')
          .attr('dominant-baseline', 'middle')
          .attr('font-size', '10px')
          .attr('font-weight', 'bold')
          .text('LoRA')
      }
    })

    // Add a note about parameter efficiency
    g.append('rect')
      .attr('x', moduleX)
      .attr('y', moduleY + modules.length * moduleSpacing + 20)
      .attr('width', moduleWidth)
      .attr('height', 60)
      .attr('rx', 5)
      .attr('fill', '#f8f9fa')
      .attr('stroke', colors.highlight)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')

    g.append('text')
      .attr('x', moduleX + 10)
      .attr('y', moduleY + modules.length * moduleSpacing + 40)
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '12px')
      .text('Parameter efficiency:')

    g.append('text')
      .attr('x', moduleX + 10)
      .attr('y', moduleY + modules.length * moduleSpacing + 60)
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text('Rank=16 vs 768 original dimension')

  }, [])

  return (
    <div className="diagram-container">
      <h2>UNet with LoRA Adapters</h2>
      <p>
        The UNet model is the core component of diffusion models like SDXL. When applying LoRA, 
        we add small adapter matrices to specific attention layers within the UNet, leaving 
        the rest of the model frozen. This diagram illustrates how LoRA adapters integrate 
        with the UNet architecture.
      </p>
      <svg ref={svgRef}></svg>
      <div className="key-points">
        <h3>Key Points</h3>
        <ul>
          <li><strong>Low-rank decomposition</strong>: LoRA decomposes weight updates into two smaller matrices (A and B)</li>
          <li><strong>Selective targeting</strong>: Only attention layers (query, key, value, output projections) typically get LoRA adapters</li>
          <li><strong>Mathematical formula</strong>: W' = W + BA × (α/r), where W is frozen, B and A are trainable, α is scaling factor, r is rank</li>
          <li><strong>Rank selection</strong>: Higher ranks (16-64) provide more capacity but require more parameters</li>
        </ul>
      </div>
    </div>
  )
}

export default UNetDiagram 