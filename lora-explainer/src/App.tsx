import { useState } from 'react'
import './App.css'
import OverviewDiagram from './components/OverviewDiagram'
import UNetDiagram from './components/UNetDiagram'
import BackpropDiagram from './components/BackpropDiagram'
import FullProcessDiagram from './components/FullProcessDiagram'
import GradientFlow from './components/interactive/GradientFlow'
import LoRAComparison from './components/interactive/LoRAComparison'

function App() {
  const [activeSection, setActiveSection] = useState(0)

  const sections = [
    { id: 0, title: 'Overview', component: <OverviewDiagram /> },
    { id: 1, title: 'UNet Forward Pass', component: <UNetDiagram /> },
    { id: 2, title: 'Backpropagation', component: <BackpropDiagram /> },
    { id: 3, title: 'Complete Process', component: <FullProcessDiagram /> },
    { id: 4, title: 'Interactive: Gradient Flow', component: <GradientFlow /> },
    { id: 5, title: 'Interactive: LoRA vs Full', component: <LoRAComparison /> },
  ]

  return (
    <div className="lora-explainer">
      <header>
        <h1>LoRA Training Process</h1>
        <p>Interactive exploration of Low-Rank Adaptation for diffusion models</p>
      </header>

      <div className="content-container">
        <nav className="sidebar">
          <ul>
            {sections.map(section => (
              <li 
                key={section.id} 
                className={activeSection === section.id ? 'active' : ''}
                onClick={() => setActiveSection(section.id)}
              >
                {section.title}
              </li>
            ))}
          </ul>
        </nav>

        <main className="content">
          {sections[activeSection].component}
        </main>
      </div>

      <footer>
        <p>Created with React, D3.js and Vite</p>
      </footer>
    </div>
  )
}

export default App
