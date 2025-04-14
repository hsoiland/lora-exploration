import React from 'react'
import { FadeIn } from '../atoms/animations/FadeIn'

const ApplicationsPage: React.FC = () => {
  // Example applications of LoRA
  const applications = [
    {
      id: 1,
      title: "Character Creation",
      description: "Train a LoRA to capture specific character features, allowing consistent generation across different prompts and settings.",
      image: "https://i.imgur.com/KPtRCQ5.png",
      bullet1: "Requires 10-20 high-quality consistent reference images",
      bullet2: "Typically uses rank 16-32 for detailed facial features",
      bullet3: "Can be combined with style LoRAs for different artistic renditions"
    },
    {
      id: 2,
      title: "Art Style Adaptation",
      description: "Adapt models to generate images in specific artistic styles, from watercolor and oil painting to anime or specific artist styles.",
      image: "https://i.imgur.com/kRtG4oM.png",
      bullet1: "Often trained with 5-15 consistent style examples",
      bullet2: "Lower ranks (4-8) can capture style effectively",
      bullet3: "Can be combined with multiple styles through weighting"
    },
    {
      id: 3,
      title: "Subject Concept Learning",
      description: "Teach models to understand specific objects, creatures, or concepts not well-represented in the base model.",
      image: "https://i.imgur.com/0cGSvb6.png",
      bullet1: "Requires varied examples of the subject (15-30 images)",
      bullet2: "Benefits from higher ranks (32-64) for complex subjects",
      bullet3: "Can be applied to both real and fictional concepts"
    },
    {
      id: 4,
      title: "Text-to-Video Adaptation",
      description: "Adapt video diffusion models with LoRA to generate videos with consistent styles or themes while maintaining temporal coherence.",
      image: "https://i.imgur.com/w7jHtx1.png",
      bullet1: "Extending image LoRAs to video generation models",
      bullet2: "Helps maintain consistency across video frames",
      bullet3: "Requires fewer examples than full video model training"
    }
  ];

  return (
    <div className="applications-page">
      <section className="page-header">
        <FadeIn>
          <h1>Applications</h1>
          <p className="header-description">
            Real-world use cases and applications of LoRA technology
          </p>
        </FadeIn>
      </section>
      
      <section className="application-examples">
        <FadeIn delay={300}>
          <h2>Popular Applications</h2>
          
          <div className="applications-grid">
            {applications.map((app, index) => (
              <div key={app.id} className="application-card">
                <div className="card-content">
                  <h3>{app.title}</h3>
                  <p>{app.description}</p>
                  <ul className="application-features">
                    <li>{app.bullet1}</li>
                    <li>{app.bullet2}</li>
                    <li>{app.bullet3}</li>
                  </ul>
                </div>
                {/* Placeholder for image - in a real implementation, use local images from assets folder */}
                <div className="card-image-placeholder" aria-label={`Example of ${app.title}`}>
                  {app.title} Example
                </div>
              </div>
            ))}
          </div>
        </FadeIn>
      </section>
      
      <section className="industry-applications">
        <FadeIn delay={600}>
          <h2>Industry Applications</h2>
          <div className="industry-grid">
            <div className="industry-card">
              <h3>Content Creation</h3>
              <p>
                Artists and designers use LoRAs to quickly adapt large models to their specific style or needs
                without retraining entire models. This enables efficient workflow for:
              </p>
              <ul>
                <li>Character design and concept art</li>
                <li>Custom illustration styles</li>
                <li>Product visualization</li>
                <li>Marketing materials with consistent branding</li>
              </ul>
            </div>
            
            <div className="industry-card">
              <h3>Entertainment</h3>
              <p>
                Game developers and animation studios leverage LoRAs to create consistent visuals
                efficiently across large projects:
              </p>
              <ul>
                <li>Game asset generation in consistent style</li>
                <li>Storyboard generation and visualization</li>
                <li>Background scene creation</li>
                <li>Character variations and expressions</li>
              </ul>
            </div>
            
            <div className="industry-card">
              <h3>Education</h3>
              <p>
                Educational platforms utilize LoRAs for creating custom visualizations tailored to specific
                domains or teaching styles:
              </p>
              <ul>
                <li>Custom educational illustrations</li>
                <li>Simplified technical diagrams</li>
                <li>Historical scene recreation</li>
                <li>Scientific visualization</li>
              </ul>
            </div>
            
            <div className="industry-card">
              <h3>Research</h3>
              <p>
                Researchers use LoRA's parameter-efficient approach for domain adaptation across different fields:
              </p>
              <ul>
                <li>Medical image analysis customization</li>
                <li>Scientific visualization of complex concepts</li>
                <li>Domain-specific model adaptation</li>
                <li>Experimental model architectures</li>
              </ul>
            </div>
          </div>
        </FadeIn>
      </section>
      
      <section className="best-practices">
        <FadeIn delay={900}>
          <h2>Best Practices for Effective LoRAs</h2>
          
          <div className="practices-container">
            <div className="practice-item">
              <h3>Consistent Training Data</h3>
              <p>
                Use consistent, high-quality images for your training set. For characters, use consistent features
                and poses. For styles, maintain consistent artistic elements across samples.
              </p>
            </div>
            
            <div className="practice-item">
              <h3>Optimal Dataset Size</h3>
              <p>
                For most cases, quality is more important than quantity. 10-20 high-quality images for characters,
                5-15 for styles, and 15-30 for diverse concepts typically yield good results.
              </p>
            </div>
            
            <div className="practice-item">
              <h3>Rank Selection</h3>
              <p>
                Match rank to your task complexity:
              </p>
              <ul>
                <li>Rank 4-8: Simple style transfer, color schemes</li>
                <li>Rank 8-16: General-purpose character or object adaptation</li>
                <li>Rank 16-32: Detailed character features or complex style elements</li>
                <li>Rank 32-64: Highly detailed concepts requiring maximum expressivity</li>
              </ul>
            </div>
            
            <div className="practice-item">
              <h3>Inference Control</h3>
              <p>
                Control LoRA influence during inference using the weight parameter. 
                Start with 0.7-0.8 and adjust based on results:
              </p>
              <ul>
                <li>Lower weights (0.3-0.6): Subtle influence</li>
                <li>Medium weights (0.7-0.9): Balanced application</li>
                <li>High weights (1.0+): Strong LoRA influence, may overpower prompt</li>
              </ul>
            </div>
          </div>
        </FadeIn>
      </section>
    </div>
  )
}

export default ApplicationsPage 