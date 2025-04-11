import { useState, useEffect, useRef } from 'react'
import * as d3 from 'd3'

const LoRAComparison = () => {
  const svgRef = useRef<SVGSVGElement>(null)
  const [rank, setRank] = useState(16)
  const [dimension, setDimension] = useState(768)
  
  useEffect(() => {
    if (!svgRef.current) return
    
    // Clear previous svg content
    d3.select(svgRef.current).selectAll('*').remove()
    
    // Calculate parameters
    const fullParams = dimension * dimension
    const loraParams = 2 * rank * dimension
    const reductionFactor = fullParams / loraParams
    const loraPercentage = (loraParams / fullParams) * 100
    
    // SVG dimensions
    const width = 800
    const height = 500
    const margin = { top: 60, right: 40, bottom: 80, left: 80 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom
    
    // Create the SVG container
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .attr('style', 'max-width: 100%; height: auto;')
    
    // Main container group with margin
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
    
    // Define data for the bar chart
    const data = [
      { type: 'Full Fine-tuning', params: fullParams, color: '#e74c3c' },
      { type: 'LoRA', params: loraParams, color: '#2ecc71' }
    ]
    
    // X scale
    const x = d3.scaleBand()
      .domain(data.map(d => d.type))
      .range([0, innerWidth])
      .padding(0.3)
    
    // Y scale (log scale for better visualization)
    const y = d3.scaleLog()
      .domain([loraParams * 0.5, fullParams * 1.5])
      .range([innerHeight, 0])
    
    // Add X axis
    g.append('g')
      .attr('transform', `translate(0, ${innerHeight})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
    
    // Add Y axis
    g.append('g')
      .call(d3.axisLeft(y)
        .tickFormat(d => {
          if (d >= 1000000) return `${(+d / 1000000).toFixed(1)}M`
          if (d >= 1000) return `${(+d / 1000).toFixed(1)}K`
          return `${+d}`
        }))
      .selectAll('text')
      .attr('font-size', '12px')
    
    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -60)
      .attr('x', -innerHeight / 2)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .text('Number of Parameters')
    
    // Create the bars
    g.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.type) || 0)
      .attr('y', d => y(d.params))
      .attr('width', x.bandwidth())
      .attr('height', d => innerHeight - y(d.params))
      .attr('fill', d => d.color)
      .attr('rx', 5)
      .attr('ry', 5)
    
    // Add parameter counts on top of bars
    g.selectAll('.bar-label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'bar-label')
      .attr('x', d => (x(d.type) || 0) + x.bandwidth() / 2)
      .attr('y', d => y(d.params) - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(d => {
        if (d.params >= 1000000) return `${(d.params / 1000000).toFixed(2)}M`
        if (d.params >= 1000) return `${(d.params / 1000).toFixed(1)}K`
        return d.params
      })
    
    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text('Parameter Comparison: Full Fine-tuning vs LoRA')
    
    // Add statistics text
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '15px')
      .attr('font-weight', 'bold')
      .text(`LoRA: ${loraPercentage.toFixed(2)}% of full model parameters (${reductionFactor.toFixed(1)}× reduction)`)
    
  }, [rank, dimension])
  
  return (
    <div className="diagram-container">
      <h2>LoRA vs Full Fine-tuning Comparison</h2>
      <p>
        Compare the number of trainable parameters between full fine-tuning and LoRA adaptation.
        Adjust the sliders to see how the parameter count changes with different matrix dimensions and LoRA ranks.
      </p>
      
      <div className="interactive-controls">
        <div className="slider-container">
          <label htmlFor="dimension-slider">Matrix Dimension:</label>
          <input
            id="dimension-slider"
            type="range"
            min={256}
            max={2048}
            step={256}
            value={dimension}
            onChange={(e) => setDimension(parseInt(e.target.value))}
          />
          <span>{dimension}×{dimension}</span>
        </div>
        
        <div className="slider-container">
          <label htmlFor="rank-slider">LoRA Rank:</label>
          <input
            id="rank-slider"
            type="range"
            min={1}
            max={64}
            step={1}
            value={rank}
            onChange={(e) => setRank(parseInt(e.target.value))}
          />
          <span>{rank}</span>
        </div>
      </div>
      
      <svg ref={svgRef}></svg>
      
      <div className="explanation">
        <h3>How It Works</h3>
        <p>
          Instead of training the full weight matrix W ∈ ℝ<sup>{dimension}×{dimension}</sup> with {(dimension * dimension).toLocaleString()} parameters,
          LoRA decomposes the update into two smaller matrices: B ∈ ℝ<sup>{dimension}×{rank}</sup> and A ∈ ℝ<sup>{rank}×{dimension}</sup>,
          with only {(2 * rank * dimension).toLocaleString()} parameters ({((2 * rank * dimension) / (dimension * dimension) * 100).toFixed(2)}% of the original).
        </p>
        <p>
          The effective weight becomes: W' = W + BA × (α/r)
        </p>
        <p>
          This dramatic reduction in trainable parameters enables efficient fine-tuning on consumer hardware while preserving model quality.
        </p>
      </div>
    </div>
  )
}

export default LoRAComparison 