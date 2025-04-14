import React from 'react'

const Footer: React.FC = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-section">
          <h4>About LoRA Explainer</h4>
          <p>
            An interactive guide to understanding Low-Rank Adaptation (LoRA) 
            techniques for efficient fine-tuning of large models.
          </p>
        </div>
        <div className="footer-section">
          <h4>Resources</h4>
          <ul>
            <li><a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noopener noreferrer">LoRA Paper</a></li>
            <li><a href="https://github.com/huggingface/peft" target="_blank" rel="noopener noreferrer">PEFT Library</a></li>
          </ul>
        </div>
      </div>
      <div className="footer-bottom">
        <p>&copy; {new Date().getFullYear()} LoRA Explainer</p>
      </div>
    </footer>
  )
}

export default Footer 