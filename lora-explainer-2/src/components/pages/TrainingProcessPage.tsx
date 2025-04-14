import React from 'react'
import { ParentSize } from '@visx/responsive'
import { FadeIn } from '../atoms/animations/FadeIn'
import TrainingProcessVisual from '../organisms/visualizations/TrainingProcessVisual'

const TrainingProcessPage: React.FC = () => {
  return (
    <div className="training-process-page">
      <section className="page-header">
        <FadeIn>
          <h1>Training Process</h1>
          <p className="header-description">
            A step-by-step visualization of the LoRA training pipeline
          </p>
        </FadeIn>
      </section>
      
      <section className="process-visualization">
        <FadeIn delay={1000}>
          <div className="visualization-container" style={{ height: 800
           }}>
            <ParentSize>
              {({ width, height }) => (
                <TrainingProcessVisual width={width} height={height} />
              )}
            </ParentSize>
          </div>
        </FadeIn>
      </section>
      
      <section className="training-details">
        <FadeIn delay={600}>
          <h2>Key Training Components</h2>
          
          <div className="training-components-grid">
            <div className="component-card">
              <h3>Frozen Base Model</h3>
              <p>
                The original model weights (billions of parameters) remain completely 
                frozen during training, preserving the general knowledge.
              </p>
            </div>
            
            <div className="component-card">
              <h3>LoRA Adapters</h3>
              <p>
                Small matrices (A and B) are injected into each layer and trained to 
                adapt the model's behavior for the specific task.
              </p>
            </div>
            
            <div className="component-card">
              <h3>Noise Prediction</h3>
              <p>
                The model learns to predict the noise added to images during the 
                diffusion process, with only the LoRA parameters being updated.
              </p>
            </div>
            
            <div className="component-card">
              <h3>Loss Function</h3>
              <p>
                Mean Squared Error (MSE) loss between predicted and actual noise
                drives the training process through gradient descent.
              </p>
            </div>
          </div>
        </FadeIn>
      </section>
      
      <section className="resource-requirements">
        <FadeIn delay={900}>
          <h2>Resource Requirements Comparison</h2>
          
          <div className="comparison-table">
            <table>
              <thead>
                <tr>
                  <th>Requirement</th>
                  <th>Full Fine-tuning</th>
                  <th>LoRA Training</th>
                  <th>Savings</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>GPU VRAM</td>
                  <td>24+ GB</td>
                  <td>6-12 GB</td>
                  <td>60-75%</td>
                </tr>
                <tr>
                  <td>Training Time</td>
                  <td>Days</td>
                  <td>Hours</td>
                  <td>70-90%</td>
                </tr>
                <tr>
                  <td>File Size</td>
                  <td>2-7 GB</td>
                  <td>5-30 MB</td>
                  <td>99%</td>
                </tr>
                <tr>
                  <td>Trainable Parameters</td>
                  <td>Billions</td>
                  <td>Millions</td>
                  <td>99%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </FadeIn>
      </section>
      
      <section className="training-tips">
        <FadeIn delay={1200}>
          <h2>Training Tips</h2>
          
          <div className="tips-list">
            <div className="tip-item">
              <h3>Optimal Dataset Size</h3>
              <p>
                For concept training, 10-30 high-quality images are typically sufficient. 
                For style training, 5-15 consistent images work well.
              </p>
            </div>
            
            <div className="tip-item">
              <h3>Learning Rate</h3>
              <p>
                Start with lower learning rates (1e-4 to 5e-4) to avoid overfitting,
                especially when working with small datasets.
              </p>
            </div>
            
            <div className="tip-item">
              <h3>Training Steps</h3>
              <p>
                Most LoRA training converges within 1000-2000 steps. 
                Watch for signs of overfitting when training longer.
              </p>
            </div>
            
            <div className="tip-item">
              <h3>Rank Selection</h3>
              <p>
                Start with rank=4 for style, rank=16 for subjects, and rank=32+ for 
                complex concepts that require more expressive power.
              </p>
            </div>
          </div>
        </FadeIn>
      </section>
    </div>
  )
}

export default TrainingProcessPage 