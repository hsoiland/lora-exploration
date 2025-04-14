import React from 'react'
import { FadeIn } from '../atoms/animations/FadeIn'
import { Group } from '@visx/group'
import { Text } from '@visx/text'
import { LinearGradient } from '@visx/gradient'
import { ParentSize } from '@visx/responsive'

const LandingPage: React.FC = () => {
  // Enhanced LoRA concept visualization
  const HeroVisualization = ({ width, height }: { width: number; height: number }) => {
    const margin = { top: 40, right: 40, bottom: 40, left: 40 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom
    
    // Dimensions for the diagram elements
    const baseModelWidth = innerWidth * 0.5
    const baseModelHeight = innerHeight * 0.6
    const baseModelX = innerWidth * 0.25
    const baseModelY = innerHeight * 0.2
    
    const loraWidth = innerWidth * 0.18
    const loraHeight = innerHeight * 0.15
    const loraX = innerWidth * 0.65
    const loraY = baseModelY + baseModelHeight * 0.2
    
    const arrowStartX = baseModelX + baseModelWidth
    const arrowStartY = baseModelY + baseModelHeight * 0.5
    const arrowEndX = loraX
    const arrowEndY = loraY + loraHeight * 0.5
    
    // Colors
    const colors = {
      baseModel: '#4361EE',
      lora: '#F72585',
      loraLight: '#F72585',
      text: '#333',
      lightText: '#666',
      arrow: '#7209B7',
      background: '#F8F9FA'
    }
    
    return (
      <svg width={width} height={height}>
        <LinearGradient 
          id="hero-gradient" 
          from="#6A11CB" 
          to="#2575FC"
          rotate={45}
        />
        
        <LinearGradient 
          id="base-model-gradient" 
          from="#4361EE" 
          to="#3A0CA3"
          vertical={true}
        />
        
        <LinearGradient 
          id="lora-gradient" 
          from="#F72585" 
          to="#7209B7"
          vertical={true}
        />
        
        <Group top={margin.top} left={margin.left}>
          {/* Background */}
          <rect
            width={innerWidth}
            height={innerHeight}
            rx={14}
            fill={colors.background}
            stroke="#EAEAEA"
            strokeWidth={1}
          />
          
          {/* Title */}
          <Text
            x={innerWidth / 2}
            y={30}
            width={innerWidth}
            verticalAnchor="middle"
            textAnchor="middle"
            fontSize={28}
            fontWeight="bold"
            fill="#2575FC"
          >
            Understanding LoRA
          </Text>
          
          {/* Base Model Box with Lock Icon */}
          <rect
            x={baseModelX}
            y={baseModelY}
            width={baseModelWidth}
            height={baseModelHeight}
            rx={8}
            fill="url(#base-model-gradient)"
            opacity={0.9}
          />
          
          {/* Lock Icon to show frozen weights */}
          <circle 
            cx={baseModelX + 30} 
            cy={baseModelY + 30}
            r={15}
            fill="white"
            opacity={0.9}
          />
          <path
            d={`M ${baseModelX + 30 - 7} ${baseModelY + 30 - 3} 
                 a 7 7 0 1 1 14 0 
                 v 4 
                 h 3 v 8 h -20 v -8 h 3 v -4`}
            fill="#3A0CA3"
          />
          
          <Text
            x={baseModelX + baseModelWidth / 2}
            y={baseModelY + 30}
            width={baseModelWidth * 0.8}
            verticalAnchor="middle"
            textAnchor="middle"
            fontSize={18}
            fontWeight="bold"
            fill="white"
          >
            Pre-trained Model
          </Text>
          
          <Text
            x={baseModelX + baseModelWidth / 2}
            y={baseModelY + 60}
            width={baseModelWidth * 0.8}
            verticalAnchor="middle"
            textAnchor="middle"
            fontSize={14}
            fill="white"
            opacity={0.9}
          >
            Weights Frozen
          </Text>
          
          {/* Internal model architecture representation */}
          <Group top={baseModelY + 90} left={baseModelX + 40}>
            {[0, 1, 2].map((layerIdx) => (
              <Group key={`layer-${layerIdx}`} top={layerIdx * 40}>
                <rect
                  x={0}
                  y={0}
                  width={baseModelWidth - 80}
                  height={30}
                  rx={4}
                  fill="rgba(255,255,255,0.2)"
                />
                <Text
                  x={10}
                  y={15}
                  verticalAnchor="middle"
                  fontSize={12}
                  fill="white"
                >
                  {layerIdx === 0 ? "Attention Layer" : 
                    layerIdx === 1 ? "Feed-Forward Layer" : "Output Layer"}
                </Text>
                
                {/* Dots to represent parameters */}
                {[0, 1, 2, 3, 4].map((dot) => (
                  <circle
                    key={`dot-${layerIdx}-${dot}`}
                    cx={baseModelWidth - 100 + dot * 12}
                    cy={15}
                    r={3}
                    fill="white"
                    opacity={0.7}
                  />
                ))}
              </Group>
            ))}
          </Group>
          
          {/* LoRA Adapter Box */}
          <rect
            x={loraX}
            y={loraY}
            width={loraWidth}
            height={loraHeight}
            rx={8}
            fill="url(#lora-gradient)"
            opacity={0.9}
          />
          
          <Text
            x={loraX + loraWidth / 2}
            y={loraY + 20}
            width={loraWidth * 0.8}
            verticalAnchor="middle"
            textAnchor="middle"
            fontSize={14}
            fontWeight="bold"
            fill="white"
          >
            LoRA Adapter
          </Text>
          
          <Text
            x={loraX + loraWidth / 2}
            y={loraY + loraHeight - 20}
            width={loraWidth * 0.8}
            verticalAnchor="middle"
            textAnchor="middle"
            fontSize={10}
            fill="white"
            opacity={0.9}
          >
            Small Trainable Weights
          </Text>
          
          {/* LoRA Rank Visual */}
          <Group top={loraY + loraHeight / 2 - 5} left={loraX + 15}>
            <Text
              x={0}
              y={0}
              fontSize={12}
              fill="white"
              opacity={0.9}
            >
              A × B
            </Text>
            
            <Text
              x={0}
              y={15}
              fontSize={10}
              fill="white"
              opacity={0.8}
            >
              r ≪ d
            </Text>
          </Group>
          
          {/* Arrow connecting base model to LoRA */}
          <path
            d={`M ${arrowStartX} ${arrowStartY} 
                 C ${arrowStartX + 40} ${arrowStartY}, 
                   ${arrowEndX - 40} ${arrowEndY}, 
                   ${arrowEndX} ${arrowEndY}`}
            stroke={colors.arrow}
            strokeWidth={2}
            fill="none"
            strokeDasharray="5,3"
            markerEnd="url(#arrow)"
          />
          
          {/* Define arrow marker */}
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="8"
              markerHeight="8"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={colors.arrow} />
            </marker>
          </defs>
          
          {/* Explanation text */}
          <Text
            x={innerWidth / 2}
            y={baseModelY + baseModelHeight + 40}
            width={innerWidth * 0.9}
            verticalAnchor="middle"
            textAnchor="middle"
            fontSize={14}
            fill={colors.text}
          >
            LoRA freezes the pre-trained model and adds small trainable adapters
          </Text>
        </Group>
      </svg>
    )
  }

  return (
    <div className="landing-page">
      <section className="hero-section py-10">
        <FadeIn delay={300} duration={1500}>
          <h1 className="text-center text-4xl font-bold text-blue-600 mb-4">Understanding LoRA</h1>
          <p className="text-center text-xl text-gray-600 mb-8">
            Low-Rank Adaptation for Efficient Fine-Tuning
          </p>
        </FadeIn>
      </section>

      <section className="intro-section mb-12">
        <FadeIn delay={600}>
          <h2>What is LoRA?</h2>
          <p>
            <strong>Low-Rank Adaptation (LoRA)</strong> is a parameter-efficient fine-tuning technique 
            that dramatically reduces the number of trainable parameters while preserving model quality.
            Instead of training all model weights, LoRA freezes the pre-trained model weights and 
            injects trainable rank decomposition matrices into each layer of the Transformer architecture.
          </p>
          <p>
            This approach allows you to fine-tune large models like Stable Diffusion with significantly 
            less memory, making it possible to train on consumer hardware. The resulting LoRA weights 
            are also much smaller (typically 1-30MB) compared to full model weights (gigabytes).
          </p>
        </FadeIn>
      </section>

      <section className="key-features mb-12">
        <FadeIn delay={900}>
          <h2>Key Benefits</h2>
          <div className="features-grid">
            <div className="feature-card">
              <h3>Efficiency</h3>
              <p>Train only ~1% of model parameters, drastically reducing memory requirements</p>
            </div>
            <div className="feature-card">
              <h3>Quality</h3>
              <p>Results comparable to full fine-tuning despite the parameter reduction</p>
            </div>
            <div className="feature-card">
              <h3>Speed</h3>
              <p>Faster training time and lower computational requirements</p>
            </div>
            <div className="feature-card">
              <h3>Portability</h3>
              <p>Small adapter files (1-30MB) rather than full models (gigabytes)</p>
            </div>
          </div>
        </FadeIn>
      </section>
      
      <section className="diagram-section mb-16">
        <FadeIn delay={1200}>
          <div className="diagram-card bg-white rounded-lg shadow-md overflow-hidden">
            <div className="diagram-container" style={{ height: '400px' }}>
              <ParentSize>
                {({ width, height }) => 
                  <HeroVisualization width={width} height={height} />
                }
              </ParentSize>
            </div>
          </div>
        </FadeIn>
      </section>
    </div>
  )
}

export default LandingPage 