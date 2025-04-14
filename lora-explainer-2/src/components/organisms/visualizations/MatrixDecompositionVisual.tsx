import React, { useState, useEffect } from 'react'
import { Group } from '@visx/group'
import { Text } from '@visx/text'
import { MarkerArrow } from '@visx/marker'
import { scaleLinear } from '@visx/scale'
import { LinearGradient } from '@visx/gradient'
import { FadeIn } from '../../atoms/animations/FadeIn'
import { cn } from '@/lib/utils'

// Custom SVG components with transitions
const TransitionRect = ({ 
  className,
  style,
  duration = 500,
  ...props 
}: React.SVGProps<SVGRectElement> & { duration?: number }) => (
  <rect
    className={cn("transition-all", className)}
    style={{ ...style, transitionDuration: `${duration}ms` }}
    {...props}
  />
)

const TransitionText = ({ 
  className,
  style,
  duration = 500,
  ...props 
}: React.SVGProps<SVGTextElement> & { duration?: number }) => (
  <text
    className={cn("transition-all", className)}
    style={{ ...style, transitionDuration: `${duration}ms` }}
    {...props}
  />
)

const TransitionPath = ({ 
  className,
  style,
  duration = 500,
  ...props 
}: React.SVGProps<SVGPathElement> & { duration?: number }) => (
  <path
    className={cn("transition-all", className)}
    style={{ ...style, transitionDuration: `${duration}ms` }}
    {...props}
  />
)

interface MatrixDecompositionVisualProps {
  width: number;
  height: number;
}

const MatrixDecompositionVisual: React.FC<MatrixDecompositionVisualProps> = ({ 
  width, 
  height 
}) => {
  // Configuration for the visualization
  const [activeStep, setActiveStep] = useState(0)
  const [dimension, setDimension] = useState(768) // Default to SDXL dimension
  const [rank, setRank] = useState(4) // Default to rank 4
  
  // Margins for the SVG
  const margin = { top: 40, left: 40, right: 40, bottom: 40 }
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom
  
  // Calculate sizes for matrices
  const matrixWidth = innerWidth * 0.25
  const matrixHeight = innerHeight * 0.5
  
  // Position for the full weight matrix W
  const wMatrixX = margin.left
  const wMatrixY = margin.top + innerHeight / 2 - matrixHeight / 2
  
  // Position for the low-rank matrices A and B
  const bMatrixX = margin.left + innerWidth * 0.5
  const bMatrixY = margin.top + innerHeight / 2 - matrixHeight / 1.5
  
  const aMatrixX = margin.left + innerWidth * 0.75
  const aMatrixY = margin.top + innerHeight / 2
  
  // Calculate parameter counts
  const fullParams = dimension * dimension
  const loraParams = rank * dimension * 2
  const reductionPercent = ((fullParams - loraParams) / fullParams * 100).toFixed(1)
  
  // Calculate dimensions based on activeStep
  const wWidth = matrixWidth
  const wHeight = matrixHeight
  
  const aWidth = matrixWidth * 0.5
  const aHeight = matrixHeight * (rank / dimension)
  
  const bWidth = matrixWidth * (rank / dimension)
  const bHeight = matrixHeight * 0.5
  
  // Handle next/previous steps
  const nextStep = () => {
    setActiveStep(prev => Math.min(prev + 1, 2))
  }
  
  const prevStep = () => {
    setActiveStep(prev => Math.max(prev - 1, 0))
  }
  
  return (
    <div className="matrix-decomposition-visual">
      <FadeIn>
        <div className="visual-controls">
          <h3>LoRA Matrix Decomposition</h3>
          <div className="controls">
            <div className="dimension-control">
              <label>Model Dimension:</label>
              <select 
                value={dimension}
                onChange={(e) => setDimension(Number(e.target.value))}
              >
                <option value={768}>768 (SD 1.5)</option>
                <option value={1024}>1024 (SDXL)</option>
                <option value={2048}>2048 (LLM Layer)</option>
              </select>
            </div>
            <div className="rank-control">
              <label>LoRA Rank:</label>
              <select 
                value={rank}
                onChange={(e) => setRank(Number(e.target.value))}
              >
                <option value={2}>2 (Smallest)</option>
                <option value={4}>4 (Compact)</option>
                <option value={16}>16 (Balanced)</option>
                <option value={32}>32 (Expressive)</option>
                <option value={64}>64 (Maximum)</option>
              </select>
            </div>
          </div>
          
          <div className="step-controls">
            <button onClick={prevStep} disabled={activeStep === 0}>Previous</button>
            <span>Step {activeStep + 1} of 3</span>
            <button onClick={nextStep} disabled={activeStep === 2}>Next</button>
          </div>
        </div>
        
        <div className="visualization-area">
          <svg width={width} height={height}>
            {/* Define gradients */}
            <LinearGradient 
              id="fullMatrix" 
              from="#3498db" 
              to="#2c3e50" 
              rotate={45}
            />
            
            <LinearGradient 
              id="loraMatrix" 
              from="#e74c3c" 
              to="#c0392b" 
              rotate={45}
            />
            
            {/* Define arrow markers */}
            <MarkerArrow 
              id="arrow" 
              refX={2} 
              size={6} 
              fill="#666" 
            />
            
            {/* Full weight matrix W */}
            <Group>
              <TransitionRect
                x={wMatrixX}
                y={wMatrixY}
                width={wWidth}
                height={wHeight}
                fill="url(#fullMatrix)"
                opacity={0.8}
                rx={4}
                duration={600}
              />
              
              <TransitionText
                x={wMatrixX + wWidth / 2}
                y={wMatrixY - 20}
                fill="#333"
                textAnchor="middle"
                fontWeight="bold"
                duration={600}
              >
                Weight Matrix W
              </TransitionText>
              
              <TransitionText
                x={wMatrixX + wWidth / 2}
                y={wMatrixY + wHeight + 30}
                fill="#666"
                textAnchor="middle"
                opacity={activeStep >= 2 ? 1 : 0}
                duration={600}
              >
                {dimension}×{dimension} = {fullParams.toLocaleString()} parameters
              </TransitionText>
            </Group>
            
            {/* Matrix B */}
            <Group>
              <TransitionRect
                x={bMatrixX}
                y={bMatrixY}
                width={bWidth}
                height={bHeight}
                fill="url(#loraMatrix)"
                opacity={activeStep >= 1 ? 0.8 : 0}
                rx={4}
                duration={600}
              />
              
              <TransitionText
                x={bMatrixX + matrixWidth * 0.15}
                y={bMatrixY - 20}
                fill="#333"
                textAnchor="middle"
                fontWeight="bold"
                opacity={activeStep >= 1 ? 1 : 0}
                duration={600}
              >
                LoRA Matrix B
              </TransitionText>
              
              <TransitionText
                x={bMatrixX + matrixWidth * 0.15}
                y={bMatrixY + bHeight + 30}
                fill="#666"
                textAnchor="middle"
                opacity={activeStep >= 2 ? 1 : 0}
                duration={600}
              >
                {dimension}×{rank} = {(dimension * rank).toLocaleString()} parameters
              </TransitionText>
            </Group>
            
            {/* Matrix A */}
            <Group>
              <TransitionRect
                x={aMatrixX}
                y={aMatrixY}
                width={aWidth}
                height={aHeight}
                fill="url(#loraMatrix)"
                opacity={activeStep >= 1 ? 0.8 : 0}
                rx={4}
                duration={600}
              />
              
              <TransitionText
                x={aMatrixX + matrixWidth * 0.15}
                y={aMatrixY - 20}
                fill="#333"
                textAnchor="middle"
                fontWeight="bold"
                opacity={activeStep >= 1 ? 1 : 0}
                duration={600}
              >
                LoRA Matrix A
              </TransitionText>
              
              <TransitionText
                x={aMatrixX + matrixWidth * 0.15}
                y={aMatrixY + aHeight + 30}
                fill="#666"
                textAnchor="middle"
                opacity={activeStep >= 2 ? 1 : 0}
                duration={600}
              >
                {rank}×{dimension} = {(dimension * rank).toLocaleString()} parameters
              </TransitionText>
            </Group>
            
            {/* Formula */}
            <TransitionText
              x={margin.left + innerWidth / 2}
              y={margin.top + 20}
              fill="#333"
              textAnchor="middle"
              fontWeight="bold"
              opacity={activeStep >= 1 ? 1 : 0}
              fontSize={18}
              duration={600}
            >
              W' = W + B×A
            </TransitionText>
            
            {/* Parameter savings */}
            <TransitionText
              x={margin.left + innerWidth / 2}
              y={margin.top + innerHeight - 20}
              fill="#e74c3c"
              textAnchor="middle"
              fontWeight="bold"
              opacity={activeStep >= 2 ? 1 : 0}
              fontSize={16}
              duration={600}
            >
              LoRA reduces parameters by {reductionPercent}%
            </TransitionText>
            
            {/* Matrix multiplication arrow */}
            <TransitionPath
              d={`M ${bMatrixX + bWidth} ${bMatrixY + bHeight / 2} 
                  L ${aMatrixX - 10} ${aMatrixY + aHeight / 2}`}
              stroke="#666"
              strokeWidth={2}
              opacity={activeStep >= 1 ? 0.8 : 0}
              markerEnd="url(#arrow)"
              fill="none"
              duration={600}
            />
            
            {/* Add arrow */}
            <TransitionPath
              d={`M ${wMatrixX + wWidth} ${wMatrixY + wHeight / 2} 
                  L ${bMatrixX - 40} ${bMatrixY + bHeight / 1.2}`}
              stroke="#666"
              strokeWidth={2}
              opacity={activeStep >= 1 ? 0.8 : 0}
              markerEnd="url(#arrow)"
              fill="none"
              duration={600}
            />
          </svg>
        </div>
        
        <div className="step-description">
          {activeStep === 0 && (
            <p>
              In traditional fine-tuning, we would update the entire weight matrix W 
              with dimensions {dimension}×{dimension}, requiring {fullParams.toLocaleString()} trainable parameters.
            </p>
          )}
          
          {activeStep === 1 && (
            <p>
              LoRA decomposes the weight update into a product of two smaller matrices: 
              B ({dimension}×{rank}) and A ({rank}×{dimension}). The effective weight becomes: W' = W + B×A
            </p>
          )}
          
          {activeStep === 2 && (
            <p>
              This reduces trainable parameters from {fullParams.toLocaleString()} to just {loraParams.toLocaleString()}, 
              a {reductionPercent}% reduction while maintaining comparable quality!
            </p>
          )}
        </div>
      </FadeIn>
    </div>
  )
}

export default MatrixDecompositionVisual 