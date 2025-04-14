import React from 'react'
import { ParentSize } from '@visx/responsive'
import { FadeIn } from '../atoms/animations/FadeIn'
import ArchitectureVisual from '../organisms/visualizations/ArchitectureVisual'

const ArchitecturePage: React.FC = () => {
  return (
    <div className="architecture-page">
      <section className="page-header">
        <FadeIn>
          <h1>Architecture Details</h1>
          <p className="header-description">
            How LoRA adapters integrate with the model architecture
          </p>
        </FadeIn>
      </section>
      
      <section className="architecture-visualization">
        <FadeIn delay={300}>
          <div className="visualization-container" style={{ height: 600 }}>
            <ParentSize>
              {({ width, height }) => (
                <ArchitectureVisual width={width} height={height} />
              )}
            </ParentSize>
          </div>
        </FadeIn>
      </section>
      
      <section className="lora-integration">
        <FadeIn delay={600}>
          <h2>How LoRA Integrates with the Model</h2>
          
          <div className="integration-details">
            <div className="detail-card">
              <h3>Target Modules</h3>
              <p>
                LoRA targets specific modules within the UNet architecture, primarily attention layers. 
                The original full-rank weights are kept frozen, and low-rank adapters are injected in 
                parallel to the original weights.
              </p>
            </div>
            
            <div className="detail-card">
              <h3>Weight Modification</h3>
              <p>
                During the forward pass, the outputs of the original frozen weights and the trainable 
                LoRA adapter matrices are added together: W' = W + BA, where W is the original weight,
                and B and A are the LoRA adapter matrices.
              </p>
            </div>
            
            <div className="detail-card">
              <h3>Implementation Method</h3>
              <p>
                Technically, LoRA is implemented by hooking into each layer's forward pass. When an input
                passes through the layer, it's processed by both the original weights and the LoRA path,
                and the results are combined.
              </p>
            </div>
          </div>
        </FadeIn>
      </section>
      
      <section className="attention-layers">
        <FadeIn delay={900}>
          <h2>Focus on Attention Layers</h2>
          
          <div className="attention-layout">
            <div className="attention-description">
              <p>
                Attention layers are the primary target for LoRA adapters because they:
              </p>
              <ul>
                <li>Heavily influence the model's understanding of content and style</li>
                <li>Contain many parameters (up to 80% in transformer-based models)</li>
                <li>Benefit most from fine-tuning for concept adaptation</li>
              </ul>
              
              <p>
                In a typical stable diffusion model, LoRA targets:
              </p>
              <ul>
                <li><strong>Query projection (Q)</strong>: Influences what the model attends to</li>
                <li><strong>Key projection (K)</strong>: Defines how content is represented</li>
                <li><strong>Value projection (V)</strong>: Determines the information being passed</li>
                <li><strong>Output projection (O)</strong>: Controls how attention affects the output</li>
              </ul>
            </div>
            
            <div className="attention-diagram">
              <FadeIn delay={1000}>
                <div className="diagram-container">
                  <pre className="code-block">
                    {`// Simplified LoRA module
function LoRALayer(x, W, A, B, alpha) {
  // Original path (frozen)
  let original = x * W; 
  
  // LoRA path (trainable)
  let lora = x * B * A * (alpha/r);
  
  // Combined result
  return original + lora;
}`}
                  </pre>
                </div>
              </FadeIn>
            </div>
          </div>
        </FadeIn>
      </section>
      
      <section className="technical-specs">
        <FadeIn delay={1200}>
          <h2>Technical Specifications</h2>
          
          <div className="specs-grid">
            <div className="spec-item">
              <h3>Rank</h3>
              <p>
                The rank parameter (r) determines the dimensionality of the low-rank approximation.
                Common values range from 4 (very efficient, less expressive) to 128 (more expressive, less efficient).
              </p>
            </div>
            
            <div className="spec-item">
              <h3>Target Modules</h3>
              <p>
                For Stable Diffusion: "q_proj", "k_proj", "v_proj", "out_proj" in cross-attention and self-attention layers.
                Optional: "to_k", "to_q", "to_v", "to_out" patterns to match different model implementations.
              </p>
            </div>
            
            <div className="spec-item">
              <h3>LoRA Alpha</h3>
              <p>
                The scaling factor (α) controls the contribution of the LoRA matrices.
                Typically set to the rank value, it can be adjusted during inference to control strength.
              </p>
            </div>
            
            <div className="spec-item">
              <h3>Network Dimensions</h3>
              <p>
                For a weight matrix W ∈ ℝ<sup>d×k</sup>, LoRA matrices are:
                B ∈ ℝ<sup>d×r</sup> and A ∈ ℝ<sup>r×k</sup>, where r ≪ min(d,k).
              </p>
            </div>
          </div>
        </FadeIn>
      </section>
    </div>
  )
}

export default ArchitecturePage 