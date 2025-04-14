import React, { useState, useEffect, useRef } from 'react'
import { Group } from '@visx/group'
import { Text } from '@visx/text'
import { Circle, Line, LinePath } from '@visx/shape'
import { LinearGradient, RadialGradient } from '@visx/gradient'
import { MarkerArrow } from '@visx/marker'
import { FadeIn } from '../../atoms/animations/FadeIn'
import { useSpring, animated, config } from 'react-spring'
import { curveBasis } from '@visx/curve'

// Animated components
const AnimatedCircle = animated(Circle);
const AnimatedLine = animated(Line);

interface ArchitectureVisualProps {
  width: number;
  height: number;
}

// Layer type definition
interface Layer {
  id: number;
  x: number;
  y: number;
  width: number;
  height: number;
  type: string;
  hasLora: boolean;
  isHighlighted: boolean;
  description: string;
}

// LoRA adapter type definition
interface LoraAdapter {
  id: number;
  layerId: number;
  x: number;
  y: number;
  width: number;
  height: number;
  isHighlighted: boolean;
  alpha: number; // LoRA scaling parameter
  rank: number;  // LoRA rank
}

// Data flow particle
interface DataParticle {
  id: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  color: string;
  speedFactor: number;
  isLoraActive: boolean;
}

// Custom rectangle component
const Rect = animated(({ 
  x, 
  y, 
  width, 
  height, 
  rx = 0, 
  fill, 
  stroke, 
  className = '',
  style = {},
  ...restProps 
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  rx?: number;
  fill?: string;
  stroke?: string;
  className?: string;
  style?: React.CSSProperties;
  [key: string]: any;
}) => (
  <rect
    x={x}
    y={y}
    width={width}
    height={height}
    rx={rx}
    fill={fill}
    stroke={stroke}
    className={className}
    style={style}
    {...restProps}
  />
));

const ArchitectureVisual: React.FC<ArchitectureVisualProps> = ({ 
  width, 
  height 
}) => {
  // State
  const showLora = true; // Always show LoRA adapters
  const [highlightLayer, setHighlightLayer] = useState<number | null>(null);
  const [particles, setParticles] = useState<DataParticle[]>([]);
  const [isAnimating, setIsAnimating] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [elapsedTime, setElapsedTime] = useState(0);
  const showExplanations = true; // Always show explanations
  const [focusedElement, setFocusedElement] = useState<string | null>(null);
  
  // Animation refs
  const animationRef = useRef<number | null>(null);
  
  // Add fullVisibility mode state
  const [fullVisibilityMode, setFullVisibilityMode] = useState(false);
  
  // Colors
  const colors = {
    primary: '#3b82f6',       // Primary blue
    secondary: '#f43f5e',     // Accent pink
    highlight: '#10b981',     // Success green
    background: '#f1f5f9',    // Light background
    text: '#1e293b',          // Dark text
    gradient1: '#4338ca',     // Gradient start (indigo)
    gradient2: '#3b82f6',     // Gradient end (blue)
    dataFlow: '#c026d3',      // Magenta for data flow
    lora: '#eab308',          // Yellow for LoRA
    loraGrad1: '#f59e0b',     // LoRA gradient start
    loraGrad2: '#d97706',     // LoRA gradient end
    layerA: '#6366f1',        // Layer type A
    layerB: '#4f46e5'         // Layer type B
  };
  
  // Animated properties
  const baseModelSpring = useSpring({
    opacity: 1,
    scale: 1,
    from: { opacity: 0, scale: 0.9 },
    config: config.gentle
  });
  
  const loraSpring = useSpring({
    opacity: showLora ? 1 : 0.4,
    scale: showLora ? 1 : 0.9,
    config: config.gentle
  });
  
  // Margins and layout dimensions
  const margin = { top: 40, right: 40, bottom: 40, left: 40 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  
  // Layer dimensions - more compact to avoid overflow
  const layerWidth = Math.min(innerWidth * 0.7, 550);
  const layerHeight = 36;
  const layerSpacing = 14;
  const totalLayers = 8;
  
  // Layer positioning - centered
  const layerX = margin.left + (innerWidth - layerWidth) / 2;
  const firstLayerY = margin.top + 70;
  
  // LoRA adapter dimensions - proportional
  const loraWidth = 70;
  const loraHeight = 36;
  const loraGap = 20;
  
  // Matrix dimensions for LoRA visualization
  const matrixWidth = 50;
  const matrixHeight = 30;
  
  // Generate layers with enhanced info
  const generateLayers = (): Layer[] => {
    const layers: Layer[] = [];
    for (let i = 0; i < totalLayers; i++) {
      const isAttentionLayer = i % 2 === 0;
      const layerY = firstLayerY + i * (layerHeight + layerSpacing);
      const layerType = isAttentionLayer ? 'Attention Layer' : 'Feed-Forward Layer';
      const hasLora = isAttentionLayer; // LoRA is typically applied to attention layers
      
      const isHighlighted = highlightLayer === i;
      
      // More detailed descriptions
      let description = '';
      if (isAttentionLayer) {
        description = "Self-attention layer with query, key, and value projections where LoRA adapters are integrated.";
      } else {
        description = "Feed-forward network with two linear transformations and a non-linearity, typically frozen during LoRA training.";
      }
      
      layers.push({
        id: i,
        x: layerX,
        y: layerY,
        width: layerWidth,
        height: layerHeight,
        type: layerType,
        hasLora,
        isHighlighted,
        description
      });
    }
    return layers;
  };
  
  // Generate LoRA adapters
  const generateLoraAdapters = (layers: Layer[]): LoraAdapter[] => {
    const adapters: LoraAdapter[] = [];
    layers.forEach(layer => {
      if (layer.hasLora) {
        adapters.push({
          id: layer.id,
          layerId: layer.id,
          x: layer.x + layer.width + loraGap,
          y: layer.y,
          width: loraWidth,
          height: loraHeight,
          isHighlighted: layer.isHighlighted,
          alpha: 0.5 + (Math.random() * 0.5), // Random alpha between 0.5-1
          rank: 4 + Math.floor(Math.random() * 4) // Random rank between 4-8
        });
      }
    });
    return adapters;
  };
  
  // Optimize particle creation with a more efficient approach
  const generateInitialParticles = (): DataParticle[] => {
    // Pre-allocate array for better performance
    const count = 25;
    const particles: DataParticle[] = new Array(count);
    
    for (let i = 0; i < count; i++) {
      // Use direct assignment instead of push
      particles[i] = createParticle(i);
    }
    return particles;
  };
  
  // Create a single particle
  const createParticle = (id: number): DataParticle => {
    // Randomize starting position to stagger particles
    const randomOffset = Math.random() * 30;
    return {
      id,
      x: margin.left - randomOffset * 2,
      y: firstLayerY + (Math.random() * 10 - 5),
      size: 2 + Math.random() * 4,
      opacity: 0.6 + Math.random() * 0.4,
      color: colors.dataFlow,
      speedFactor: 0.7 + Math.random() * 0.6,
      isLoraActive: false
    };
  };
  
  // Reset animation and state with improved cleanup
  const resetAnimation = () => {
    // Ensure animation is stopped completely
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    setIsAnimating(false);
    
    // Reset all state values to initial state
    setElapsedTime(0);
    setHighlightLayer(null);
    setFocusedElement(null);
    
    // Reset particles with fresh generation
    setParticles(generateInitialParticles());
    
    // Delay animation start slightly to ensure state updates are applied
    setTimeout(() => {
      setIsAnimating(true);
      startAnimation();
    }, 50);
  };
  
  // Animation control functions
  const startAnimation = () => {
    if (animationRef.current !== null) return;
    setIsAnimating(true);
    let lastTime = 0;

    resetAnimation();
    
    const animate = (time: number) => {
      if (lastTime === 0) {
        lastTime = time;
        animationRef.current = requestAnimationFrame(animate);
        return;
      }
      
      const delta = (time - lastTime) * animationSpeed * 0.001;
      lastTime = time;
      
      // Update elapsed time for cycling effects
      setElapsedTime(prev => prev + delta);
      
      // Update particles
      setParticles(prevParticles => {
        // Create a new array directly rather than mapping
        const newParticles = new Array(prevParticles.length);
        
        for (let idx = 0; idx < prevParticles.length; idx++) {
          const particle = prevParticles[idx];
          
          // These calculations remain as before but are more efficiently structured
          let newX = particle.x;
          let newY = particle.y;
          let newColor = particle.color;
          let newOpacity = particle.opacity;
          let isLoraActive = particle.isLoraActive;
          
          // Get current layer based on y position
          const layerIndex = Math.floor((newY - firstLayerY) / (layerHeight + layerSpacing));
          const isInLoraLayer = layerIndex >= 0 && layerIndex < totalLayers && layerIndex % 2 === 0;
          
          // Before first layer - moving right
          if (newX < layerX) {
            newX += delta * 50 * particle.speedFactor * animationSpeed;
          }
          // Inside a layer - moving right
          else if (newX < layerX + layerWidth) {
            newX += delta * 40 * particle.speedFactor * animationSpeed;
            
            // Small vertical oscillation for visual appeal
            newY += Math.sin(elapsedTime * 10 + particle.id) * delta * 5;
          }
          // After a layer - determine path
          else if (newX >= layerX + layerWidth && newX < layerX + layerWidth + 5) {
            // If in a LoRA layer and LoRA is active, send to LoRA
            if (isInLoraLayer && showLora) {
              // 80% chance to go to LoRA, 20% chance to go to next layer
              if (Math.random() < 0.8) {
                isLoraActive = true;
                // Start moving to LoRA
                newX = layerX + layerWidth + (delta * 30 * particle.speedFactor * animationSpeed);
                // Adjust color to indicate LoRA pathway
                newColor = colors.lora;
              } else {
                // Go to next layer
                newY = firstLayerY + (layerIndex + 1) * (layerHeight + layerSpacing) + layerHeight / 2;
                newX = layerX + layerWidth - 5; // Move slightly back to create curve
              }
            } else {
              // Go to next layer
              if (layerIndex < totalLayers - 1) {
                newY = firstLayerY + (layerIndex + 1) * (layerHeight + layerSpacing) + layerHeight / 2;
                newX = layerX - 5; // Move to left side of next layer
              }
            }
          }
          // Between layer and LoRA - moving to LoRA
          else if (isLoraActive && newX >= layerX + layerWidth + 5 && newX < layerX + layerWidth + loraGap) {
            newX += delta * 30 * particle.speedFactor * animationSpeed;
          }
          // Inside LoRA - process
          else if (isLoraActive && newX >= layerX + layerWidth + loraGap && newX < layerX + layerWidth + loraGap + loraWidth) {
            // Move through LoRA
            newX += delta * 20 * particle.speedFactor * animationSpeed;
            
            // Pulsate in LoRA
            newOpacity = Math.max(0.2, Math.min(0.9, 0.5 + Math.sin(elapsedTime * 5 + particle.id * 3) * 0.3));
            
            // Size fluctuation in LoRA
            particle.size = Math.max(2, Math.min(6, 4 + Math.sin(elapsedTime * 3 + particle.id * 2) * 2));
          }
          // After LoRA - return to main flow
          else if (isLoraActive && newX >= layerX + layerWidth + loraGap + loraWidth) {
            // Find the next layer
            if (layerIndex < totalLayers - 1) {
              // Return to main flow - curve down to next layer
              newY = lerp(
                newY, 
                firstLayerY + (layerIndex + 1) * (layerHeight + layerSpacing) + layerHeight / 2, 
                delta * 3
              );
              newX = lerp(newX, layerX - 10, delta * 3 * particle.speedFactor * animationSpeed);
              
              // Reset color when returning to main flow
              if (Math.random() < 0.1) {
                newColor = colors.dataFlow;
              }
            } else {
              // Last layer - exit the visual
              newX += delta * 30 * particle.speedFactor * animationSpeed;
              // Fade out
              newOpacity = Math.max(0, newOpacity - delta * 0.5);
            }
          }
          // Moving between layers (not in a layer or LoRA)
          else if (newX < layerX) {
            // Move right to enter layer
            newX += delta * 40 * particle.speedFactor * animationSpeed;
          }
          
          // Reset particles that complete the flow or go off-screen
          if (
            newX > width - margin.right || 
            newY > height - margin.bottom ||
            newY < margin.top ||
            (layerIndex >= totalLayers && newX > layerX + layerWidth + 50)
          ) {
            newParticles[idx] = createParticle(particle.id);
            continue;
          }
          
          // Directly create new object instead of using spread operator
          newParticles[idx] = {
            id: particle.id,
            x: newX,
            y: newY,
            size: particle.size,
            opacity: newOpacity,
            color: newColor,
            speedFactor: particle.speedFactor,
            isLoraActive
          };
        }
        
        return newParticles;
      });
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
  };
  
  const stopAnimation = () => {
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    setIsAnimating(false);
  };
  
  // Helper function for linear interpolation
  const lerp = (a: number, b: number, t: number) => {
    return a + (b - a) * t;
  };

  
  const handleLayerHover = (layerId: number | null) => {
    setHighlightLayer(layerId);
  };
  
  const handleElementFocus = (element: string | null) => {
    setFocusedElement(element === focusedElement ? null : element);
  };
  
  // Check if an element is active for styling
  const isElementActive = (element: string): boolean => {
    if (focusedElement) {
      return focusedElement === element;
    }
    if (element === 'baseModel') {
      return highlightLayer === null;
    }
    if (element.startsWith('layer-')) {
      const layerId = parseInt(element.split('-')[1], 10);
      return highlightLayer === layerId;
    }
    if (element.startsWith('lora-')) {
      const layerId = parseInt(element.split('-')[1], 10);
      return highlightLayer === layerId && showLora;
    }
    return false;
  };
  
  // Toggle full visibility mode - shows all elements at once with dramatic lighting effects
  const toggleFullVisibilityMode = () => {
    // Toggle the state
    setFullVisibilityMode(!fullVisibilityMode);
    
    if (!fullVisibilityMode) {
      // Activating full visibility - with dramatic "light up" effect
      // Ensure animation is running
      if (!isAnimating) {
        setIsAnimating(true);
      }
      
      // Clear any focused elements to show everything
      setHighlightLayer(null);
      setFocusedElement(null);
      
      // Create more particles for a richer visual experience
      const enhancedParticles: DataParticle[] = [];
      for (let i = 0; i < 45; i++) { // Significantly more particles
        const particle = createParticle(i);
        // Make particles more vibrant and larger
        particle.size = 3 + Math.random() * 5;
        particle.opacity = 0.7 + Math.random() * 0.3;
        enhancedParticles.push(particle);
      }
      setParticles(enhancedParticles);
      
      // Speed up animation for more visual impact
      setAnimationSpeed(2.0);
      
      // Start animation with enhanced settings
      stopAnimation(); // First stop existing animation
      setTimeout(() => {
        startAnimation();
      }, 50);
    } else {
      // Returning to normal mode
      resetAnimation();
    }
  };
  
  // Jump to final state with full animation effect
  const jumpToFinalState = () => {
    // If already in full visibility mode, do nothing
    if (fullVisibilityMode) return;
    
    // Call toggleFullVisibilityMode to activate the full animation effect
    toggleFullVisibilityMode();
  };
  
  // Update getElementOpacity function to add glow and increased vibrance in full visibility mode
  const getElementOpacity = (element: string): number => {
    // In full visibility mode, keep everything visible with increased vibrance
    if (fullVisibilityMode) {
      return 1.0; // Full opacity for everything
    }
    
    if (focusedElement) {
      return focusedElement === element ? 1 : 0.3;
    }
    
    if (element === 'lora' && !showLora) {
      return 0.3;
    }
    
    return isElementActive(element) ? 1 : 0.7;
  };
  
  // Add a new function to handle the enhanced visual styling for elements in full visibility mode
  const getElementStyle = (element: string): React.CSSProperties => {
    if (!fullVisibilityMode) {
      return isElementActive(element) ? 
        { filter: 'drop-shadow(0px 0px 5px rgba(255,255,255,0.7))' } : {};
    }
    
    // Enhanced visual style when in full visibility mode
    return {
      filter: 'drop-shadow(0px 0px 10px rgba(255,255,255,0.9))',
      transition: 'all 0.5s ease-in-out',
      animation: `pulse-${element} 2s infinite ease-in-out`
    };
  };
  
  // Generate descriptions for layers and components with enhanced content for full visibility mode
  const getDescriptionFor = (component: string): string => {
    // If in full visibility mode, provide enhanced descriptions
    if (fullVisibilityMode) {
      if (component === 'baseModel') {
        return "Stable Diffusion XL UNet model with all parameters frozen, preserving foundational capabilities";
      }
      
      if (component === 'lora') {
        return "Fully activated LoRA modules efficiently fine-tune model with only 0.1-1% of the base parameters";
      }
      
      if (component.startsWith('layer-')) {
        const layerId = parseInt(component.split('-')[1], 10);
        if (layerId % 2 === 0) {
          return "Attention layer with LoRA adapters integrated into Q/K/V projections for efficient fine-tuning";
        } else {
          return "Feed-forward layer remains frozen while attention layers are optimized with LoRA adapters";
        }
      }
      
      if (component.startsWith('lora-')) {
        return "LoRA adapters use rank decomposition (A×B matrices) to dramatically reduce trainable parameters";
      }
      
      return "Complete SDXL UNet architecture with all LoRA adapters active and efficiently fine-tuned";
    }
    
    // Standard descriptions for normal mode
    if (component === 'baseModel') {
      return "Pre-trained UNet model with frozen parameters during LoRA training";
    }
    
    if (component === 'lora') {
      return "Low-Rank Adaptation modules that enable efficient fine-tuning with minimal parameters";
    }
    
    if (component.startsWith('layer-')) {
      const layerId = parseInt(component.split('-')[1], 10);
      const layers = generateLayers();
      const layer = layers.find(l => l.id === layerId);
      return layer?.description || "Model layer";
    }
    
    if (component.startsWith('lora-')) {
      const layerId = parseInt(component.split('-')[1], 10);
      if (layerId % 2 === 0) {
        return "LoRA adapter for attention projections (Q, K, V matrices) with low-rank decomposition";
      } else {
        return "LoRA adapter for feed-forward projections";
      }
    }
    
    return "SDXL UNet architecture with LoRA adaptation modules";
  };
  
  // Add a simple performance optimization by memoizing layers and adapters generation
  const layers = React.useMemo(() => generateLayers(), [highlightLayer]);
  const loraAdapters = React.useMemo(() => generateLoraAdapters(layers), [layers]);
  
  // Ensure animation resets completely on mount with no dependencies
  useEffect(() => {
    // Use setTimeout to ensure component is fully mounted before animation starts
    const timer = setTimeout(() => {
      // Reset animation completely - this prepares initial state
      resetAnimation();
    }, 100);
    
    return () => {
      clearTimeout(timer);
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, []);
  
  // Streamline the matrix rendering to be more efficient
  const renderLoraMatrix = (adapter: LoraAdapter) => {
    const { x, y, width, height, isHighlighted, rank } = adapter;
    
    // Matrix elements
    const matrixElements: React.ReactNode[] = [];
    
    // Combined matrix visualization instead of separate A and B
    const matrixRows = rank;
    const matrixCols = 6;
    const cellSize = 3;
    const padding = 1;
    const matrixX = x + 10;
    const matrixY = y + 15;
    
    // Single loop to create all matrix cells in one pass
    for (let i = 0; i < matrixRows; i++) {
      for (let j = 0; j < matrixCols; j++) {
        // Left side is matrix A (blue tint)
        if (j < matrixCols/2) {
          const value = Math.sin(elapsedTime * 2 + i + j) * 0.5 + 0.5;
          const intensity = Math.floor(value * 255).toString(16).padStart(2, '0');
          const color = `#${intensity}${intensity}ff`;
          
          matrixElements.push(
            <rect
              key={`matrixA-${adapter.id}-${i}-${j}`}
              x={matrixX + j * (cellSize + padding)}
              y={matrixY + i * (cellSize + padding)}
              width={cellSize}
              height={cellSize}
              fill={color}
              opacity={isHighlighted ? 0.9 : 0.6}
            />
          );
        } 
        // Right side is matrix B (red tint)
        else {
          const value = Math.cos(elapsedTime * 2.5 + i + j) * 0.5 + 0.5;
          const intensity = Math.floor(value * 255).toString(16).padStart(2, '0');
          const color = `#ff${intensity}${intensity}`;
          
          matrixElements.push(
            <rect
              key={`matrixB-${adapter.id}-${i}-${j}`}
              x={matrixX + width - 30 + (j-matrixCols/2) * (cellSize + padding)}
              y={matrixY + i * (cellSize + padding)}
              width={cellSize}
              height={cellSize}
              fill={color}
              opacity={isHighlighted ? 0.9 : 0.6}
            />
          );
        }
      }
    }
    
    // Multiplication symbol
    matrixElements.push(
      <text
        key={`matrix-mul-${adapter.id}`}
        x={x + width / 2}
        y={matrixY + 8}
        fontSize={8}
        fill="white"
        textAnchor="middle"
        opacity={isHighlighted ? 0.9 : 0.6}
      >
        ×
      </text>
    );
    
    return matrixElements;
  };
  
  // Render explanations based on current focus with enhanced styling for full visibility mode
  const renderExplanations = () => {
    if (!showExplanations) return null;
    
    let explanation = "";
    
    if (fullVisibilityMode) {
      explanation = "Complete SDXL UNet with active LoRA adapters efficiently fine-tuned on specific layers";
    } else if (focusedElement) {
      explanation = getDescriptionFor(focusedElement);
    } else if (highlightLayer !== null) {
      const layer = layers.find(l => l.id === highlightLayer);
      if (layer) {
        explanation = layer.description;
      }
    } else {
      explanation = "SDXL UNet architecture with LoRA modules integrates small, trainable adapters into key layers";
    }
    
    return (
      <g className="explanation-box">
        <rect
          x={margin.left + 20}
          y={margin.top - 25}
          width={innerWidth - 40}
          height={35}
          rx={5}
          fill={fullVisibilityMode ? "rgba(30, 30, 60, 0.85)" : "rgba(0, 0, 0, 0.8)"}
          opacity={0.9}
          style={fullVisibilityMode ? {
            animation: 'pulse 2s infinite ease-in-out',
            filter: 'drop-shadow(0px 0px 8px rgba(255,255,255,0.4))'
          } : {}}
        />
        <text
          x={margin.left + innerWidth / 2}
          y={margin.top - 8}
          fontSize={fullVisibilityMode ? 14 : 13}
          fill={fullVisibilityMode ? "#ffd700" : "white"}
          fontWeight={fullVisibilityMode ? "bold" : "normal"}
          textAnchor="middle"
          dominantBaseline="middle"
        >
          {explanation}
        </text>
      </g>
    );
  };
  
  // Render technical details at the bottom
  const renderTechnicalDetails = () => {
    if (!showExplanations) return null;
    
    // Get current focus or highlighted layer
    let techDetails = "";
    
    if (focusedElement === 'lora' || (highlightLayer !== null && layers[highlightLayer]?.hasLora)) {
      techDetails = "LoRA: W + ΔW where ΔW = A×B, reduces parameters from d×d to r×(2d) where r << d";
    } else if (focusedElement === 'baseModel' || highlightLayer === null) {
      techDetails = "UNet2DConditionModel with frozen weights (requires_grad=False) to preserve general capabilities";
    } else if (highlightLayer !== null) {
      const layer = layers[highlightLayer];
      if (layer.type.includes('Attention')) {
        techDetails = "MultiHeadAttention(q_proj, k_proj, v_proj, out_proj) with LoRA applied to projection matrices";
      } else {
        techDetails = "FeedForward(fc1, act, fc2) transformations with weights kept frozen during training";
      }
    }
    
    return (
      <g className="tech-details">
        <rect
          x={margin.left + 20}
          y={height - margin.bottom - 20}
          width={innerWidth - 40}
          height={30}
          rx={5}
          fill="rgba(40, 40, 80, 0.9)"
          opacity={0.9}
        />
        <text
          x={margin.left + innerWidth / 2}
          y={height - margin.bottom - 5}
          fontSize={11}
          fill="white"
          textAnchor="middle"
          dominantBaseline="middle"
          fontFamily="monospace"
        >
          {techDetails}
        </text>
      </g>
    );
  };
  
  // Add these CSS keyframes to the top of the component (inside the component function but before return):
  useEffect(() => {
    // Create and inject CSS for animations when component mounts
    if (typeof document !== 'undefined') {
      const style = document.createElement('style');
      style.type = 'text/css';
      style.innerHTML = `
        @keyframes pulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.05); }
          100% { transform: scale(1); }
        }
        
        @keyframes pulse-lora {
          0% { transform: scale(1); filter: drop-shadow(0px 0px 5px rgba(255,220,100,0.6)); }
          50% { transform: scale(1.08); filter: drop-shadow(0px 0px 15px rgba(255,220,100,0.9)); }
          100% { transform: scale(1); filter: drop-shadow(0px 0px 5px rgba(255,220,100,0.6)); }
        }
      `;
      document.head.appendChild(style);
      
      return () => {
        document.head.removeChild(style);
      };
    }
  }, []);
  
  return (
    <div className="architecture-visual">
      <FadeIn>
        <div className="visual-controls mb-4 bg-slate-100 p-4 rounded-lg">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-lg font-semibold text-slate-800">LoRA Architecture Integration</h3>
              <p className="text-sm text-slate-600">
                Low-rank adapters apply efficient fine-tuning to specific layers
              </p>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={jumpToFinalState}
                className="px-3 py-1.5 rounded-md bg-indigo-500 text-white"
                title="Show the final stage with smooth animation"
              >
                Final Stage
              </button>
              <button 
                onClick={toggleFullVisibilityMode}
                className={`px-3 py-1.5 rounded-md ${fullVisibilityMode ? 'bg-green-500 text-white' : 'bg-purple-500 text-white'}`}
                title="Show all components with enhanced animated effects"
              >
                {fullVisibilityMode ? 'Normal Mode' : 'Full Animation'}
              </button>
              {isAnimating ? (
                <button 
                  onClick={stopAnimation}
                  className="px-3 py-1.5 rounded-md bg-red-500 text-white"
                >
                  Pause
                </button>
              ) : (
                <button 
                  onClick={startAnimation}
                  className="px-3 py-1.5 rounded-md bg-blue-500 text-white"
                >
                  Play
                </button>
              )}
              <button 
                onClick={resetAnimation}
                className="px-3 py-1.5 rounded-md bg-slate-300 text-slate-700"
              >
                Reset
              </button>
            </div>
          </div>
        </div>
        
        <div className="bg-white border rounded-lg shadow-sm overflow-hidden">
          <svg 
            width={width} 
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Define gradients */}
            <LinearGradient
              id="baseModelGradient"
              from={colors.gradient1}
              to={colors.gradient2}
              vertical={true}
            />
            
            <LinearGradient
              id="loraGradient"
              from={colors.loraGrad1}
              to={colors.loraGrad2}
              vertical={true}
            />
            
            <LinearGradient 
              id="layerAGradient" 
              from={colors.layerA} 
              to={colors.gradient2}
              vertical={true}
            />
            
            <LinearGradient 
              id="layerBGradient" 
              from={colors.layerB} 
              to={colors.gradient1}
              vertical={true}
            />
            
            <RadialGradient
              id="glowGradient"
              from={colors.lora}
              to="transparent"
            />
            
            <MarkerArrow
              id="arrow"
              fill={colors.dataFlow}
              refX={2}
              size={6}
            />
            
            {/* Title and subtitle */}
            <Group>
              <Text
                x={width / 2}
                y={margin.top - 10}
                textAnchor="middle"
                verticalAnchor="start"
                fontSize={18}
                fontWeight="bold"
                fill={colors.text}
              >
                SDXL UNet with LoRA Integration
              </Text>
            </Group>
          
            {/* Main visualization group */}
            <Group top={0} left={0}>
              {/* Background */}
              <rect
                width={width}
                height={height}
                fill={colors.background}
                opacity={0.3}
                rx={10}
              />
              
              {/* Render explanations */}
              {renderExplanations()}
              
              {/* Render technical details */}
              {renderTechnicalDetails()}
              
              {/* Model layers */}
              {layers.map((layer, index) => (
                <Group 
                  key={`layer-${layer.id}`}
                  onClick={() => handleElementFocus(`layer-${layer.id}`)}
                  onMouseEnter={() => handleLayerHover(layer.id)}
                  onMouseLeave={() => handleLayerHover(null)}
                  style={{ cursor: 'pointer' }}
                >
                  {/* Main layer box */}
                  <Rect
                    x={layer.x}
                    y={layer.y}
                    width={layer.width}
                    height={layer.height}
                    rx={6}
                    fill={layer.type.includes('Attention') ? 'url(#layerAGradient)' : 'url(#layerBGradient)'}
                    opacity={layer.isHighlighted ? 0.9 : getElementOpacity(`layer-${layer.id}`)}
                    stroke={layer.isHighlighted ? colors.highlight : (layer.type.includes('Attention') ? colors.layerA : colors.layerB)}
                    strokeWidth={layer.isHighlighted || fullVisibilityMode ? 2 : 1}
                    style={{
                      filter: fullVisibilityMode ? 
                        `drop-shadow(0px 0px ${8 + Math.sin(elapsedTime * 3 + layer.id) * 4}px rgba(255,255,255,0.7))` : 
                        (layer.isHighlighted ? 'drop-shadow(0px 0px 4px rgba(255,255,255,0.5))' : 'none'),
                      transition: 'filter 0.3s ease, opacity 0.3s ease',
                      animation: fullVisibilityMode ? `pulse 1.5s infinite ease-in-out` : 'none'
                    }}
                  />
                  
                  {/* Layer label */}
                  <Text
                    x={layer.x + 20}
                    y={layer.y + layer.height / 2}
                    verticalAnchor="middle"
                    fill="white"
                    fontSize={12}
                    fontWeight={layer.isHighlighted ? 'bold' : 'normal'}
                    opacity={layer.isHighlighted ? 1 : 0.9}
                  >
                    {layer.type}
                  </Text>
                  
                  {/* Layer indicators for attention */}
                  {layer.type.includes('Attention') && (
                    <>
                      <g className="attention-indicators">
                        {[0, 1, 2, 3].map(headIndex => (
                          <circle
                            key={`attention-head-${layer.id}-${headIndex}`}
                            cx={layer.x + layer.width - 60 + headIndex * 12}
                            cy={layer.y + layer.height / 2}
                            r={3}
                            fill="white"
                            opacity={0.7 + Math.sin(elapsedTime * 2 + headIndex + layer.id) * 0.3}
                          />
                        ))}
                      </g>
                      
                      {/* Advanced detailed view when highlighted */}
                      {layer.isHighlighted && (
                        <g className="detailed-attention-view">
                          {[0, 1, 2].map(row => 
                            <g key={`attention-detail-${layer.id}-${row}`}>
                              <line 
                                x1={layer.x + 40} 
                                y1={layer.y + 10 + row * 8}
                                x2={layer.x + layer.width - 40}
                                y2={layer.y + 10 + row * 8}
                                stroke="rgba(255,255,255,0.3)"
                                strokeWidth={1}
                                strokeDasharray="1,2"
                              />
                              
                              {[0, 1, 2, 3].map(col => (
                                <circle
                                  key={`attention-matrix-${layer.id}-${row}-${col}`}
                                  cx={layer.x + 60 + col * 25}
                                  cy={layer.y + 10 + row * 8}
                                  r={2 + Math.sin(elapsedTime * 3 + row + col) * 1}
                                  fill="white"
                                  opacity={0.3 + Math.sin(elapsedTime * 2 + row * col) * 0.3}
                                />
                              ))}
                            </g>
                          )}
                          
                          <text
                            x={layer.x + 30}
                            y={layer.y + 8}
                            fontSize={6}
                            fill="white"
                            opacity={0.8}
                          >
                            Q/K/V Projections
                          </text>
                        </g>
                      )}
                    </>
                  )}
                  
                  {/* Connection to next layer */}
                  {index < layers.length - 1 && (
                    <AnimatedLine
                      from={{ x: layer.x + layer.width / 2, y: layer.y + layer.height }}
                      to={{ x: layer.x + layer.width / 2, y: layer.y + layer.height + layerSpacing }}
                      stroke={colors.dataFlow}
                      strokeWidth={1.5}
                      opacity={0.5}
                      strokeDasharray="4,4"
                      markerEnd="url(#arrow)"
                      style={{
                        opacity: getElementOpacity(`layer-${layer.id}`)
                      }}
                    />
                  )}
                </Group>
              ))}
              
              {/* LoRA adapters */}
              {showLora && loraAdapters.map((adapter) => (
                <Group 
                  key={`lora-adapter-${adapter.id}`}
                  onClick={() => handleElementFocus(`lora-${adapter.id}`)}
                  onMouseEnter={() => handleLayerHover(adapter.layerId)}
                  onMouseLeave={() => handleLayerHover(null)}
                  style={{ cursor: 'pointer' }}
                >
                  {/* LoRA module background */}
                  <Rect
                    x={adapter.x}
                    y={adapter.y}
                    width={adapter.width}
                    height={adapter.height}
                    rx={6}
                    fill="url(#loraGradient)"
                    opacity={adapter.isHighlighted ? 0.9 : getElementOpacity(`lora-${adapter.id}`)}
                    stroke={colors.lora}
                    strokeWidth={adapter.isHighlighted || fullVisibilityMode ? 2 : 1}
                    style={{
                      filter: fullVisibilityMode ? 
                        `drop-shadow(0px 0px ${10 + Math.sin(elapsedTime * 4 + adapter.id) * 5}px rgba(255,220,100,0.8))` : 
                        (adapter.isHighlighted ? 'drop-shadow(0px 0px 4px rgba(255,255,255,0.5))' : 'none'),
                      transition: 'filter 0.3s ease, opacity 0.3s ease',
                      animation: fullVisibilityMode ? `pulse-lora 2s infinite ease-in-out` : 'none'
                    }}
                  />
                  
                  {/* LoRA label */}
                  <Text
                    x={adapter.x + adapter.width / 2}
                    y={adapter.y + 12}
                    textAnchor="middle"
                    verticalAnchor="middle"
                    fill="white"
                    fontSize={11}
                    fontWeight={adapter.isHighlighted ? 'bold' : 'normal'}
                  >
                    {`LoRA r=${adapter.rank}`}
                  </Text>
                  
                  {/* LoRA matrices visualization */}
                  {renderLoraMatrix(adapter)}
                  
                  {/* Connection from layer to LoRA */}
                  <AnimatedLine
                    from={{ x: layerX + layerWidth, y: adapter.y + adapter.height / 2 }}
                    to={{ x: adapter.x, y: adapter.y + adapter.height / 2 }}
                    stroke={colors.lora}
                    strokeWidth={adapter.isHighlighted ? 2 : 1.5}
                    opacity={adapter.isHighlighted ? 0.9 : 0.6}
                    strokeDasharray={adapter.isHighlighted ? "0" : "3,3"}
                    style={{
                      opacity: getElementOpacity(`lora-${adapter.id}`) * 0.8
                    }}
                  />
                </Group>
              ))}
              
              {/* Data flow particles */}
              {particles.map(particle => (
                <circle
                  key={`particle-${particle.id}`}
                  cx={particle.x}
                  cy={particle.y}
                  r={particle.size}
                  fill={particle.color}
                  opacity={particle.opacity}
                />
              ))}
            </Group>
            
            {/* Legend */}
            <Group top={height - margin.bottom - 70} left={margin.left + 20}>
              <rect
                x={0}
                y={0}
                width={14}
                height={14}
                rx={3}
                fill="url(#layerAGradient)"
                opacity={0.7}
              />
              <text
                x={20}
                y={7}
                dominantBaseline="middle"
                fill={colors.text}
                fontSize={12}
              >
                Attention Layer (LoRA Applied)
              </text>
              
              <rect
                x={0}
                y={20}
                width={14}
                height={14}
                rx={3}
                fill="url(#layerBGradient)"
                opacity={0.7}
              />
              <text
                x={20}
                y={27}
                dominantBaseline="middle"
                fill={colors.text}
                fontSize={12}
              >
                Feed-Forward Layer (Frozen)
              </text>
              
              <rect
                x={0}
                y={40}
                width={14}
                height={14}
                rx={3}
                fill="url(#loraGradient)"
                opacity={0.7}
              />
              <text
                x={20}
                y={47}
                dominantBaseline="middle"
                fill={colors.text}
                fontSize={12}
              >
                LoRA Adapter (Trainable)
              </text>
              
              <circle
                cx={7}
                cy={67}
                r={4}
                fill={colors.dataFlow}
                opacity={0.7}
              />
              <text
                x={20}
                y={67}
                dominantBaseline="middle"
                fill={colors.text}
                fontSize={12}
              >
                Data Flow
              </text>
            </Group>
          </svg>
        </div>
        
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">SDXL LoRA Architecture Benefits</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-50 border p-4 rounded-lg">
              <h4 className="font-medium text-slate-800">Memory Efficient</h4>
              <p className="text-sm text-slate-600">Adapters are &lt;1% the size of the full model</p>
            </div>
            <div className="bg-slate-50 border p-4 rounded-lg">
              <h4 className="font-medium text-slate-800">Fast Training</h4>
              <p className="text-sm text-slate-600">Train in minutes on consumer GPUs</p>
            </div>
            <div className="bg-slate-50 border p-4 rounded-lg">
              <h4 className="font-medium text-slate-800">Preserves Base Knowledge</h4>
              <p className="text-sm text-slate-600">Frozen base prevents catastrophic forgetting</p>
            </div>
            <div className="bg-slate-50 border p-4 rounded-lg">
              <h4 className="font-medium text-slate-800">Composable</h4>
              <p className="text-sm text-slate-600">Multiple LoRAs can be combined at inference time</p>
            </div>
          </div>
        </div>
      </FadeIn>
    </div>
  );
};

export default ArchitectureVisual; 