import { useState } from 'react'
import './App.css'

// Import our components
import Header from './components/molecules/Header'
import Footer from './components/molecules/Footer'
import LandingPage from './components/pages/LandingPage'
import LoraBasicsPage from './components/pages/LoraBasicsPage'
import TrainingProcessPage from './components/pages/TrainingProcessPage'
import ArchitecturePage from './components/pages/ArchitecturePage'
import ApplicationsPage from './components/pages/ApplicationsPage'
import AdvancedTopicsPage from './components/pages/AdvancedTopicsPage'

function App() {
  const [activePage, setActivePage] = useState('home')

  // Render the active page based on the state
  const renderActivePage = () => {
    switch (activePage) {
      case 'home':
        return <LandingPage />
      case 'basics':
        return <LoraBasicsPage />
      case 'training':
        return <TrainingProcessPage />
      case 'architecture':
        return <ArchitecturePage />
      case 'applications':
        return <ApplicationsPage />
      case 'advanced':
        return <AdvancedTopicsPage />
      default:
        return <LandingPage />
    }
  }

  return (
    <div className="lora-explainer-app">
      <Header setActivePage={setActivePage} activePage={activePage} />
      <main className="content-container">
        {renderActivePage()}
      </main>
      <Footer />
    </div>
  )
}

export default App
