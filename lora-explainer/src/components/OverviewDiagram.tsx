import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

const OverviewDiagram = () => {
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

    // Main container group with margin
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Define the training steps
    const steps = [
      { id: 1, name: "Load Image & Caption", y: 50, description: "Start with high-quality images and captions" },
      { id: 2, name: "Encode to Latent", y: 120, description: "VAE compresses image to latent representation" },
      { id: 3, name: "Add Noise", y: 190, description: "Random noise is added based on timestep" },
      { id: 4, name: "UNet Forward Pass", y: 260, description: "Predict noise using frozen weights + LoRA adapters" },
      { id: 5, name: "Compare Prediction", y: 330, description: "Compare predicted noise with original noise" },
      { id: 6, name: "Calculate Loss", y: 400, description: "Mean Squared Error between prediction and target" },
      { id: 7, name: "Backpropagate", y: 470, description: "Gradients flow back through the network" },
      { id: 8, name: "Update LoRA Only", y: 540, description: "Only small adapter matrices are modified" }
    ]

    // Define training process colors
    const colors = {
      box: '#e6f7ff',
      activeBox: '#3498db',
      line: '#95a5a6',
      text: '#2c3e50',
      accent: '#e74c3c'
    }

    // Draw connector lines between steps
    for (let i = 0; i < steps.length - 1; i++) {
      g.append('line')
        .attr('x1', width / 2)
        .attr('y1', steps[i].y + 30)
        .attr('x2', width / 2)
        .attr('y2', steps[i + 1].y - 10)
        .attr('stroke', colors.line)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
    }

    // Create groups for each step
    const stepGroups = g.selectAll('.step')
      .data(steps)
      .enter()
      .append('g')
      .attr('class', 'step')
      .attr('transform', d => `translate(${width/2 - 150}, ${d.y})`)

    // Add step number circles
    stepGroups.append('circle')
      .attr('cx', -30)
      .attr('cy', 15)
      .attr('r', 15)
      .attr('fill', colors.activeBox)
      .attr('stroke', 'white')
      .attr('stroke-width', 2)

    // Add step numbers
    stepGroups.append('text')
      .attr('x', -30)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text(d => d.id)

    // Add step boxes
    stepGroups.append('rect')
      .attr('width', 300)
      .attr('height', 40)
      .attr('rx', 5)
      .attr('ry', 5)
      .attr('fill', (d, i) => i === 7 ? '#ffeaa7' : colors.box)
      .attr('stroke', (d, i) => i === 7 ? colors.accent : colors.activeBox)
      .attr('stroke-width', 2)

    // Add step names
    stepGroups.append('text')
      .attr('x', 150)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-weight', 'bold')
      .attr('fill', colors.text)
      .text(d => d.name)

    // Add descriptions
    stepGroups.append('text')
      .attr('x', 330)
      .attr('y', 20)
      .attr('text-anchor', 'start')
      .attr('dominant-baseline', 'middle')
      .attr('fill', colors.text)
      .attr('font-style', 'italic')
      .text(d => d.description)

    // Highlight LoRA-specific step
    g.append('text')
      .attr('x', width / 2 + 200)
      .attr('y', 560)
      .attr('text-anchor', 'middle')
      .attr('font-weight', 'bold')
      .attr('fill', colors.accent)
      .text('Only ~1% of weights are trained!')

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '20px')
      .attr('font-weight', 'bold')
      .text('LoRA Training Process Overview')

  }, [])

  return (
    <div className="diagram-container">
      <h2>LoRA Training Process Overview</h2>
      <p>
        LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that dramatically 
        reduces the number of trainable parameters while preserving model quality. The diagram below
        shows the key steps in the LoRA training process for diffusion models.
      </p>
      <svg ref={svgRef}></svg>
      <div className="key-points">
        <h3>Key Benefits</h3>
        <ul>
          <li><strong>Efficiency</strong>: Train only ~1% of the model's parameters</li>
          <li><strong>Quality</strong>: Achieve results comparable to full fine-tuning</li>
          <li><strong>Speed</strong>: Faster training time and lower memory requirements</li>
          <li><strong>Portability</strong>: Small adapter files (typically 1-30MB)</li>
        </ul>
      </div>
    </div>
  )
}

export default OverviewDiagram 