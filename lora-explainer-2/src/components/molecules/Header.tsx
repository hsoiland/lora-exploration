import React from 'react'

interface HeaderProps {
  activePage: string;
  setActivePage: (page: string) => void;
}

const Header: React.FC<HeaderProps> = ({ activePage, setActivePage }) => {
  // Navigation items
  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'basics', label: 'LoRA Basics' },
    { id: 'training', label: 'Training Process' },
    { id: 'architecture', label: 'Architecture' },
    { id: 'applications', label: 'Applications' },
    { id: 'advanced', label: 'Advanced Topics' },
  ]

  return (
    <header className="header">
      <div className="logo">
        <h1>LoRA Explainer</h1>
      </div>
      <nav className="main-nav">
        <ul>
          {navItems.map(item => (
            <li key={item.id}>
              <button 
                className={`nav-item ${activePage === item.id ? 'active' : ''}`}
                onClick={() => setActivePage(item.id)}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  )
}

export default Header 