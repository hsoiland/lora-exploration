import React from 'react'
import { ParentSize } from '@visx/responsive'
import { FadeIn } from '../atoms/animations/FadeIn'
import MatrixDecompositionVisual from '../organisms/visualizations/MatrixDecompositionVisual'

const LoraBasicsPage: React.FC = () => {
  return (
    <div className="lora-basics-page">
      <section className="page-header">
        <FadeIn>
          <h1>LoRA Basics</h1>
          <p className="header-description">
            Understanding the fundamental concepts behind Low-Rank Adaptation
          </p>
        </FadeIn>
      </section>
      
      <section className="concept-introduction">
        <FadeIn delay={300}>
          <h2>The Problem: Fine-tuning is Inefficient</h2>
          <p>
            Traditional fine-tuning of large models like Stable Diffusion requires updating 
            all model weights, which is computationally expensive and memory-intensive. For models 
            with billions of parameters, this presents significant challenges:
          </p>
          
          <ul className="problem-list">
            <li>High GPU memory requirements (24+ GB for SDXL)</li>
            <li>Long training times</li>
            <li>Large storage needs for each fine-tuned version</li>
            <li>Difficulty in sharing and distributing fine-tuned models</li>
          </ul>
        </FadeIn>
      </section>
      
      <section className="lora-solution">
        <FadeIn delay={600}>
          <h2>The Solution: Low-Rank Decomposition</h2>
          <p>
            LoRA (Low-Rank Adaptation) addresses these issues with a key insight: <strong>weight changes 
            during fine-tuning have a low "intrinsic rank"</strong>. This means we can represent these changes 
            using much smaller matrices without losing model quality.
          </p>
          
          <div className="visualization-container" style={{ height: 600 }}>
            <ParentSize>
              {({ width, height }) => (
                <MatrixDecompositionVisual width={width} height={height} />
              )}
            </ParentSize>
          </div>
          
          <div className="explanation-box">
            <h3>Key Formula</h3>
            <p className="formula">W' = W + B×A×(α/r)</p>
            <p>Where:</p>
            <ul>
              <li><strong>W</strong>: Original weight matrix (frozen during training)</li>
              <li><strong>W'</strong>: Effective weight matrix after LoRA adaptation</li>
              <li><strong>B</strong>: Low-rank update matrix (dimension × rank)</li>
              <li><strong>A</strong>: Low-rank update matrix (rank × dimension)</li>
              <li><strong>α</strong>: Scaling factor (typically 1.0)</li>
              <li><strong>r</strong>: The rank parameter (smaller = fewer parameters, less expressive)</li>
            </ul>
          </div>
        </FadeIn>
      </section>
      
      <section className="practical-benefits">
        <FadeIn delay={900}>
          <h2>Practical Benefits</h2>
          
          <div className="benefits-grid">
            <div className="benefit-card">
              <h3>Memory Efficiency</h3>
              <p>
                Train with 80-95% less memory by only updating the low-rank matrices
                instead of all weights.
              </p>
            </div>
            
            <div className="benefit-card">
              <h3>Faster Training</h3>
              <p>
                Reduced parameter count leads to quicker training cycles and iterations.
              </p>
            </div>
            
            <div className="benefit-card">
              <h3>Small File Size</h3>
              <p>
                LoRA adapters are typically 1-30MB compared to 2-7GB for full models.
              </p>
            </div>
            
            <div className="benefit-card">
              <h3>Composability</h3>
              <p>
                Multiple LoRAs can be combined and balanced to create new effects.
              </p>
            </div>
          </div>
        </FadeIn>
      </section>
      
      <section className="technical-considerations">
        <FadeIn delay={1200}>
          <h2>Technical Considerations</h2>
          
          <div className="considerations-list">
            <div className="consideration-item">
              <h3>Choosing the Rank</h3>
              <p>
                Higher rank values provide more expressivity but require more parameters:
              </p>
              <ul>
                <li><strong>Rank 2-4:</strong> Minimal adaptation, best for subtle style changes</li>
                <li><strong>Rank 8-16:</strong> Balanced for most use cases</li>
                <li><strong>Rank 32-64:</strong> Higher expressivity for complex concepts</li>
              </ul>
            </div>
            
            <div className="consideration-item">
              <h3>Target Modules</h3>
              <p>
                Not all layers need LoRA adaptation. Typically applied to:
              </p>
              <ul>
                <li>Attention layers (query, key, value projections)</li>
                <li>Cross-attention for text-to-image conditioning</li>
                <li>Optionally some feed-forward layers</li>
              </ul>
            </div>
            
            <div className="consideration-item">
              <h3>Alpha Parameter</h3>
              <p>
                The alpha scaling factor (α) controls the influence of the LoRA adaptation. 
                Common practice is to set α equal to the rank r, then scale during inference.
              </p>
            </div>
          </div>
        </FadeIn>
      </section>
    </div>
  )
}

export default LoraBasicsPage 