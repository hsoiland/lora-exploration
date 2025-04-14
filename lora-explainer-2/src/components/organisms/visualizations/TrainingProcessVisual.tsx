import React, { useState, useEffect, useRef } from 'react'
import { Group } from '@visx/group'
import { Text } from '@visx/text'
import { Circle, Line, LinePath } from '@visx/shape'
import { LinearGradient, RadialGradient } from '@visx/gradient'
import { MarkerArrow, MarkerCross } from '@visx/marker'
import { FadeIn } from '../../atoms/animations/FadeIn'
import { cn } from '@/lib/utils'
import { useSpring, animated, config } from 'react-spring'
import { curveBasis } from '@visx/curve'

// Animated components
const AnimatedCircle = animated(Circle);
const AnimatedLinePath = animated(LinePath);

interface TrainingProcessVisualProps {
  width: number;
  height: number;
}

// Points for flowing data
interface DataPoint {
  id: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  color: string;
  speedFactor: number;
}

/**
 * Redesigned Training Process Visualization
 * 
 * Uses continuous animation to show the LoRA training flow instead of
 * step-by-step navigation. Data flows naturally through the pipeline.
 */
const TrainingProcessVisual: React.FC<TrainingProcessVisualProps> = ({ 
  width, 
  height 
}) => {
  // Animation state
  const [isAnimating, setIsAnimating] = useState(true); // Start animation by default
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const animationRef = useRef<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [predictionAccuracy, setPredictionAccuracy] = useState(0);
  const [lossValue, setLossValue] = useState(1);
  const [loraActive, setLoraActive] = useState(false);
  const [matrixUpdateVisible, setMatrixUpdateVisible] = useState(false);
  const [showExplanations, setShowExplanations] = useState(true);
  
  // Focus state to highlight sections when clicked
  const [focusedElement, setFocusedElement] = useState<string | null>(null);
  
  // Add full visibility mode
  const [fullVisibilityMode, setFullVisibilityMode] = useState(false);
  
  // Animated properties for components
  const baseModelSpring = useSpring({
    opacity: 1,
    scale: 1,
    from: { opacity: 0, scale: 0.9 },
    config: config.gentle
  });
  
  const loraSpring = useSpring({
    opacity: loraActive ? 1 : 0.4,
    scale: loraActive ? 1 : 0.9,
    glow: loraActive ? 10 : 0,
    config: config.gentle
  });
  
  const noiseSpring = useSpring({
    intensity: noiseLevel,
    config: { tension: 80, friction: 10 }
  });
  
  const predictionSpring = useSpring({
    accuracy: predictionAccuracy,
    config: { tension: 60, friction: 10 }
  });
  
  const lossSpring = useSpring({
    value: lossValue,
    config: { tension: 40, friction: 10 }
  });
  
  // Define the training process steps
  const trainingSteps = [
    {
      id: 0,
      name: "Initial State",
      description: "Before training begins, the base model is frozen and the LoRA adapters are initialized",
      focusArea: "baseModel"
    },
    {
      id: 1,
      name: "Data Input",
      description: "Training data flows through the frozen base model for initial processing",
      focusArea: "dataFlow"
    },
    {
      id: 2,
      name: "Noise Addition",
      description: "Controlled noise is added to help the model learn the denoising process",
      focusArea: "noise"
    },
    {
      id: 3,
      name: "Prediction",
      description: "The model predicts the original input by removing noise, with LoRA adapters activated",
      focusArea: "prediction"
    },
    {
      id: 4,
      name: "Loss Calculation",
      description: "The difference between prediction and target is measured as loss",
      focusArea: "loss"
    },
    {
      id: 5,
      name: "Parameter Update",
      description: "Only the LoRA parameters are updated based on the loss, not the base model",
      focusArea: "lora"
    },
    {
      id: 6,
      name: "Feedback Loop",
      description: "Updated parameters flow back to influence the next training iteration",
      focusArea: "feedback"
    },
    {
      id: 7,
      name: "Complete Training",
      description: "LoRA training complete - all mechanisms active with optimized parameters",
      focusArea: "complete"
    }
  ];
  
  // Auto-progress through the training stages
  useEffect(() => {
    // Auto progress based on elapsed time
    const stageTime = 5; // Seconds per stage
    
    // Calculate the current stage index
    let currentStageIndex = Math.min(
      Math.floor(elapsedTime / stageTime),
      trainingSteps.length - 1
    );
    
    // Special case: If we're in the last step (handling edge case with timing)
    // This ensures final step is properly recognized
    if (elapsedTime >= (trainingSteps.length - 1) * stageTime) {
      currentStageIndex = trainingSteps.length - 1; // Force to final step
    }
    
    // Calculate stage transition percentage (0-1)
    const stageProgress = (elapsedTime % stageTime) / stageTime;
    
    // Update visual properties based on current stage
    updateVisualState(currentStageIndex, stageProgress);
    
  }, [elapsedTime, trainingSteps.length]);
  
  // Update visual state based on current stage and transition progress
  const updateVisualState = (stageIndex: number, progress: number) => {
    // Current and next stage (for transitions)
    const currentStage = trainingSteps[stageIndex];
    const nextStage = stageIndex < trainingSteps.length - 1 
      ? trainingSteps[stageIndex + 1] 
      : null;
    
    // Set appropriate visual properties based on the current stage
    switch(stageIndex) {
      case 0: // Initial state
        setNoiseLevel(progress * 0.2);
        setPredictionAccuracy(progress * 0.2);
        setLossValue(1);
        setLoraActive(progress > 0.5);
        setMatrixUpdateVisible(false);
        break;
      case 1: // Data input
        setNoiseLevel(0.2 + progress * 0.6);
        setPredictionAccuracy(0.2 + progress * 0.1);
        setLossValue(1 - progress * 0.1);
        setLoraActive(true);
        setMatrixUpdateVisible(false);
        break;
      case 2: // Noise addition
        setNoiseLevel(0.8 - progress * 0.2);
        setPredictionAccuracy(0.3 + progress * 0.4);
        setLossValue(0.9 - progress * 0.2);
        setLoraActive(true);
        setMatrixUpdateVisible(false);
        break;
      case 3: // Prediction
        setNoiseLevel(0.6 - progress * 0.1);
        setPredictionAccuracy(0.7 + progress * 0.1);
        setLossValue(0.7 - progress * 0.2);
        setLoraActive(true);
        setMatrixUpdateVisible(false);
        break;
      case 4: // Loss calculation
        setNoiseLevel(0.5 - progress * 0.1);
        setPredictionAccuracy(0.8 + progress * 0.05);
        setLossValue(0.5 - progress * 0.2);
        setLoraActive(true);
        setMatrixUpdateVisible(progress > 0.5);
        break;
      case 5: // Parameter update
        setNoiseLevel(0.4 - progress * 0.1);
        setPredictionAccuracy(0.85 + progress * 0.05);
        setLossValue(0.3 - progress * 0.1);
        setLoraActive(true);
        setMatrixUpdateVisible(true);
        break;
      case 6: // Feedback loop
        setNoiseLevel(0.3 - progress * 0.2);
        setPredictionAccuracy(0.9 + progress * 0.05);
        setLossValue(0.2 - progress * 0.1);
        setLoraActive(true);
        setMatrixUpdateVisible(true);
        break;
      case 7: // Complete training
        setNoiseLevel(0.7);
        setPredictionAccuracy(0.98);
        setLossValue(0.1);
        setLoraActive(true);
        setMatrixUpdateVisible(true);
        break;
    }
  };
  
  // Get current focus area based on focused element or current stage in the animation
  const getCurrentFocusArea = () => {
    // If in full visibility mode, return "complete" to indicate final state
    if (fullVisibilityMode) return "complete";
    
    if (focusedElement) return focusedElement;
    
    // Check if we're at the final step directly
    const stageTime = 5; // Same as elsewhere
    const finalStepTime = stageTime * (trainingSteps.length - 1);
    
    if (elapsedTime >= finalStepTime) {
      return "complete"; // Use the same final area as full visibility mode
    }
    
    // Auto focus based on elapsed time for other steps
    const currentStageIndex = Math.floor(elapsedTime / stageTime);
    
    // Ensure stage index is valid
    if (currentStageIndex >= 0 && currentStageIndex < trainingSteps.length) {
      return trainingSteps[currentStageIndex].focusArea;
    }
    
    return "baseModel"; // Default fallback
  };

  // Handle click on a visualization element
  const handleElementClick = (element: string) => {
    // If already focused on this element, remove focus
    if (focusedElement === element) {
      setFocusedElement(null);
    } else {
      setFocusedElement(element);
    }
  };

  // Animation control functions
  const startAnimation = () => {
    if (animationRef.current !== null) return;
    setIsAnimating(true);
    let lastTime = 0;
    
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
      
      // Update data points with improved movement logic
      setDataPoints(prevPoints => {
        return prevPoints.map(point => {
          // Determine current path and position
          let newX = point.x;
          let newY = point.y;
          let newColor = point.color;
          let newOpacity = point.opacity;
          
          // Input path - smoother entry
          if (point.x < baseModelX - 20) {
            // Linear movement to the left of the base model
            newX += delta * 50 * point.speedFactor * animationSpeed;
            
            // Ensure points stay on the horizontal input path
            newY = lerp(point.y, innerHeight * 0.23, delta * 5);
          }
          // Through model path
          else if (point.x < baseModelX + baseModelWidth) {
            // Move through the model at consistent speed
            newX += delta * 40 * point.speedFactor * animationSpeed;
            
            // Keep y-position stable
            newY = innerHeight * 0.23 + (Math.sin(point.x * 0.1) * 5); // Slight sine wave motion inside model
          }
          // Exit model to prediction path
          else if (point.x < baseModelX + baseModelWidth + 50) {
            // Continue movement after exiting the model
            newX += delta * 35 * point.speedFactor * animationSpeed;
            
            // Start curving toward throughModelPath[2]
            const targetX = throughModelPath[2].x;
            const targetY = throughModelPath[2].y;
            const t = (point.x - (baseModelX + baseModelWidth)) / 50;
            newY = lerp(innerHeight * 0.23, targetY, Math.min(1, t));
          }
          // To noise/prediction area
          else if (point.y < innerHeight * 0.51) {
            // Use the throughModelPath for guidance - move to prediction area
            const tx = point.x / innerWidth; // Normalized x for curve shaping
            
            // Follow curve more precisely with bezier-like interpolation
            newX = lerp(
              point.x, 
              throughModelPath[2].x + (Math.sin(point.id * 5) * 10), 
              delta * (1.5 + tx) * animationSpeed
            );
            newY = lerp(
              point.y, 
              throughModelPath[2].y + (Math.cos(point.id * 3) * 8),  
              delta * (1.5 + tx) * animationSpeed
            );
            
            // Add noise with smarter positioning
            if (point.y > innerHeight * 0.28 && point.color !== colors.noise && Math.random() < 0.03) {
              newColor = colors.noise;
              // Make noise particles slightly larger and more visible
              newOpacity = 0.7 + Math.random() * 0.3;
            }
          }
          // To loss calculation
          else if (point.y < innerHeight * 0.58) {
            // Follow the prediction path curve
            newX = lerp(point.x, predictionPath[1].x + (Math.sin(point.id * 3) * 8), delta * 2 * animationSpeed);
            newY = lerp(point.y, predictionPath[1].y + (Math.cos(point.id * 2) * 5), delta * 2 * animationSpeed);
            
            // If near prediction endpoint, move toward loss
            if (Math.abs(point.x - predictionPath[1].x) < 20 && Math.abs(point.y - predictionPath[1].y) < 20) {
              newX = lerp(point.x, predictionPath[2].x, delta * 3 * animationSpeed);
              newY = lerp(point.y, predictionPath[2].y, delta * 3 * animationSpeed);
            }
            
            // Change to prediction color with higher probability if it's noise
            if (point.color === colors.noise && Math.random() < 0.15) {
              newColor = colors.prediction;
            }
          }
          // To LoRA (from loss)
          else if (point.x < loraX && Math.abs(point.y - innerHeight * 0.58) < 15) {
            // Follow loss path to LoRA
            newX = lerp(point.x, lossPath[2].x, delta * 3 * animationSpeed);
            newY = lerp(point.y, lossPath[2].y, delta * 3 * animationSpeed);
            
            // Change to loss color near the circle
            if (Math.abs(point.x - predictionPath[2].x) < 30 && 
                Math.abs(point.y - predictionPath[2].y) < 30 && 
                Math.random() < 0.2) {
              newColor = colors.secondary;
            }
          }
          // Inside LoRA - move down through adapter
          else if (point.x >= loraX && point.x <= loraX + loraWidth && 
                  point.y >= loraY && point.y < loraY + loraHeight) {
            // Move down through the LoRA adapter
            newY += delta * 30 * point.speedFactor * animationSpeed;
            
            // Add slight horizontal movement
            newX += (Math.sin(elapsedTime * 5 + point.id) * delta * 5);
            
            // Constrain within LoRA boundaries
            newX = Math.max(loraX + 5, Math.min(loraX + loraWidth - 5, newX));
            
            // Change to LoRA color
            if (Math.random() < 0.1) {
              newColor = colors.lora;
            }
          }
          // LoRA feedback loop start
          else if (point.y >= loraY + loraHeight && point.y < loraY + loraHeight + 20 && 
                  point.x >= loraX && point.x <= loraX + loraWidth) {
            // Initial downward movement from LoRA
            newY = lerp(point.y, loraFeedbackPath[2].y, delta * 3 * animationSpeed);
            newX = lerp(point.x, loraFeedbackPath[2].x, delta * 3 * animationSpeed);
          }
          // LoRA feedback middle section
          else if (point.y >= innerHeight * 0.6) {
            // Handle the horizontal part of the feedback path
            if (point.x > innerWidth * 0.45) {
              // Moving leftward
              newX = lerp(point.x, loraFeedbackPath[3].x, delta * 3 * animationSpeed);
              newY = lerp(point.y, loraFeedbackPath[3].y, delta * 1.5 * animationSpeed);
            } 
            // Handle leftward curve and upward movement
            else {
              // Moving up toward model
              newX = lerp(point.x, baseModelX - 15, delta * 2 * animationSpeed);
              newY = lerp(point.y, innerHeight * 0.4, delta * 3 * animationSpeed);
            }
            
            // Ensure points follow the feedback path
            if (Math.random() < 0.1) {
              newColor = colors.lora;
            }
          }
          // Back to model - final approach
          else if (point.y < innerHeight * 0.45 && point.y > innerHeight * 0.3 && point.x < baseModelX) {
            // Final approach to re-enter the model
            newX = lerp(point.x, baseModelX, delta * 4 * animationSpeed);
            newY = lerp(point.y, innerHeight * 0.33, delta * 3 * animationSpeed);
          }
          
          // Reset point if it completes the loop or goes out of bounds
          if (
            (point.x > baseModelX && point.x < baseModelX + 10 && point.y < innerHeight * 0.35 && point.y > innerHeight * 0.2) ||
            point.x > innerWidth - margin.right / 2 ||
            point.y > innerHeight - margin.bottom / 2
          ) {
            return createDataPoint(point.id);
          }
          
          return {
            ...point,
            x: newX,
            y: newY,
            color: newColor,
            opacity: newOpacity
          };
        });
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
  
  const resetAnimation = () => {
    // Ensure animation is fully stopped first
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    setIsAnimating(false);
    
    // Reset all state values to initial values
    setElapsedTime(0);
    setNoiseLevel(0);
    setPredictionAccuracy(0);
    setLossValue(1);
    setLoraActive(false);
    setMatrixUpdateVisible(false);
    setFocusedElement(null);
    
    // Reset data points with more points for smoother flow
    const initialPoints: DataPoint[] = [];
    for (let i = 0; i < 25; i++) {
      initialPoints.push(createDataPoint(i));
    }
    setDataPoints(initialPoints);
    
    // Delay starting animation to ensure state updates are applied
    setTimeout(() => {
      setIsAnimating(true);
      startAnimation();
    }, 100); // Use slightly longer timeout to ensure all state is applied
  };

  // Get current stage description based on elapsed time or visibility mode
  const getCurrentStageDescription = () => {
    // If in full visibility mode, always show final step description
    if (fullVisibilityMode) {
      return "Final Step: Training complete - all components active with optimized LoRA parameters";
    }
    
    // Check if we're at the final step directly by comparing elapsed time
    const stageTime = 5; // Seconds per stage  
    const finalStepTime = stageTime * (trainingSteps.length - 1);
    
    if (elapsedTime >= finalStepTime) {
      return "Final Step: Training complete - all components active with optimized LoRA parameters";
    }
    
    // For other steps, calculate based on stage index
    const currentStageIndex = Math.floor(elapsedTime / stageTime);
    
    // Return appropriate step description based on stage index
    switch (currentStageIndex) {
      case 0:
        return "Step 1: Data is fed through the frozen base model for initial processing";
      case 1:
        return "Step 2: Noise prediction begins with help from LoRA adapters";
      case 2: 
        return "Step 3: Controlled noise is added to help the model learn denoising";
      case 3:
        return "Step 4: Model predicts original input by removing noise";
      case 4:
        return "Step 5: Loss is calculated by comparing prediction with target";
      case 5:
        return "Step 6: Only LoRA parameters are updated, base model remains frozen";
      case 6:
        return "Step 7: Updated parameters flow back to influence the next training iteration";
      default:
        return "LoRA training process visualization";
    }
  };

  // Function to directly jump to the final training step with full animation
  const jumpToFinalStep = () => {
    // If we're already in full visibility mode, do nothing
    if (fullVisibilityMode) return;
    
    // Simply call toggleFullVisibilityMode to activate full animation
    toggleFullVisibilityMode();
  };

  // Modify the toggleFullVisibilityMode function to link better with the step progression
  const toggleFullVisibilityMode = () => {
    // Toggle the state
    setFullVisibilityMode(!fullVisibilityMode);
    
    if (!fullVisibilityMode) {
      // First navigate to final step for smooth transition
      // Set elapsed time to reach the final step
      const stageTime = 5; // Same as elsewhere
      setElapsedTime(stageTime * 7); // Jump to step 7
      
      // Now activate full visibility with dramatic effects
      if (!isAnimating) {
        setIsAnimating(true);
      }
      
      // Set all visual elements to optimal values with maximum impact
      setNoiseLevel(0.7);  // Higher noise for more dramatic effect
      setPredictionAccuracy(0.98);  // Near-perfect prediction accuracy
      setLossValue(0.1);  // Very low loss (excellent training)
      setLoraActive(true);  // LoRA fully active
      setMatrixUpdateVisible(true);  // Show matrix updates
      setFocusedElement(null);  // No focus to show everything
      
      // Speed up animation for more visual excitement
      setAnimationSpeed(2.0);
      
      // Create many more particles for a much richer visual experience
      const enhancedPoints: DataPoint[] = [];
      for (let i = 0; i < 60; i++) { // Significantly more particles 
        const particle = createDataPoint(i);
        // Make particles more vibrant and diverse
        particle.size = 3 + Math.random() * 6;
        particle.opacity = 0.75 + Math.random() * 0.25;
        particle.speedFactor = 0.9 + Math.random() * 0.8; // Faster particles
        
        // Diversify particle colors
        if (Math.random() < 0.3) {
          particle.color = colors.lora;
        } else if (Math.random() < 0.4) {
          particle.color = colors.prediction;
        }
        
        enhancedPoints.push(particle);
      }
      setDataPoints(enhancedPoints);
      
      // Stop any existing animation to reset
      stopAnimation();
      
      // Restart animation with enhanced settings
      setTimeout(() => {
        startAnimation();
      }, 50);
    } else {
      // Returning to normal mode
      resetAnimation();
    }
  };

  // Add CSS animations for the full visibility mode - right after the toggleFullVisibilityMode function
  useEffect(() => {
    // Create and inject CSS for the "light up" animations
    if (typeof document !== 'undefined') {
      const style = document.createElement('style');
      style.type = 'text/css';
      style.innerHTML = `
        @keyframes pulse-glow {
          0% { filter: drop-shadow(0px 0px 5px rgba(255,255,255,0.6)); }
          50% { filter: drop-shadow(0px 0px 15px rgba(255,255,255,0.9)); }
          100% { filter: drop-shadow(0px 0px 5px rgba(255,255,255,0.6)); }
        }
        
        @keyframes pulse-lora {
          0% { transform: scale(1); filter: drop-shadow(0px 0px 8px rgba(255,215,0,0.7)); }
          50% { transform: scale(1.08); filter: drop-shadow(0px 0px 20px rgba(255,215,0,0.9)); }
          100% { transform: scale(1); filter: drop-shadow(0px 0px 8px rgba(255,215,0,0.7)); }
        }
        
        @keyframes pulse-model {
          0% { transform: scale(1); }
          50% { transform: scale(1.03); }
          100% { transform: scale(1); }
        }
        
        @keyframes flash-matrix {
          0%, 100% { opacity: 0.8; }
          50% { opacity: 1; }
        }
      `;
      document.head.appendChild(style);
      
      return () => {
        document.head.removeChild(style);
      };
    }
  }, []);

  // Is a particular element active in the current step?
  const isElementActive = (element: string) => {
    if (focusedElement) {
      return focusedElement === element;
    }
    
    const focusArea = getCurrentFocusArea();
    
    switch(element) {
      case 'baseModel':
        return focusArea === 'baseModel';
      case 'dataFlow':
        return focusArea === 'dataFlow' || focusArea === 'baseModel';
      case 'noise':
        return focusArea === 'noise';
      case 'prediction':
        return focusArea === 'prediction';
      case 'loss':
        return focusArea === 'loss';
      case 'lora':
        return focusArea === 'lora' || focusArea === 'feedback';
      case 'feedback':
        return focusArea === 'feedback';
      default:
        return false;
    }
  };
  
  // Modify getElementOpacity to respect full visibility mode
  const getElementOpacity = (element: string) => {
    // Special case for data points that might have various colors
    if (element.startsWith('#')) {
      if (element === colors.dataFlow) return getElementOpacity('dataFlow');
      if (element === colors.noise) return getElementOpacity('noise');
      if (element === colors.prediction) return getElementOpacity('prediction');
      if (element === colors.secondary) return getElementOpacity('loss');
      if (element === colors.lora) return getElementOpacity('lora');
      return 0.7; // Default opacity for unidentified colors
    }
    
    // In full visibility mode, all elements are fully visible
    if (fullVisibilityMode) {
      return 1.0;
    }
    
    if (focusedElement) {
      return focusedElement === element ? 1 : 0.25;
    }
    return isElementActive(element) ? 1 : 0.4;
  };

  // Get a highlight style for the current step's focus area
  const getHighlightStyle = (element: string) => {
    // Enhanced styling for full visibility (final) mode 
    if (fullVisibilityMode) {
      switch (element) {
        case 'baseModel':
          return {
            filter: 'drop-shadow(0px 0px 15px rgba(80,110,250,0.8))',
            animation: 'pulse-model 3s infinite ease-in-out',
            transition: 'all 0.5s ease-in-out',
          };
        case 'lora':
          return {
            filter: 'drop-shadow(0px 0px 20px rgba(255,215,0,0.8))',
            animation: 'pulse-lora 2s infinite ease-in-out',
            transition: 'all 0.5s ease-in-out',
          };
        case 'loss':
          return {
            filter: 'drop-shadow(0px 0px 12px rgba(244,63,94,0.8))',
            animation: 'pulse-glow 2.5s infinite ease-in-out',
            transition: 'all 0.5s ease-in-out',
          };
        case 'prediction':
          return {
            filter: 'drop-shadow(0px 0px 15px rgba(16,185,129,0.8))',
            animation: 'pulse-glow 2.2s infinite ease-in-out',
            transition: 'all 0.5s ease-in-out',
          };
        default:
          return {
            filter: 'drop-shadow(0px 0px 10px rgba(255,255,255,0.7))',
            animation: 'pulse-glow 3s infinite ease-in-out',
            transition: 'all 0.5s ease-in-out',
          };
      }
    }
    
    // Original styling for regular mode
    if (isElementActive(element)) {
      return {
        filter: 'drop-shadow(0px 0px 5px rgba(255,255,255,0.7))',
        transition: 'filter 0.5s ease-in-out',
      };
    }
    return {};
  };

  // Get specific explanation text for each component based on elapsed time or mode
  const getExplanationFor = (component: string): string => {
    // If in full visibility mode, return enhanced descriptions
    if (fullVisibilityMode) {
      switch(component) {
        case 'baseModel':
          return "Pre-trained foundation model remains frozen throughout training, preserving base capabilities";
        case 'lora':
          return "Optimized LoRA adapters provide efficient fine-tuning with only 0.1%-1% of base model parameters";
        case 'noise':
          return "Controlled noise training enhances stability and generalization during diffusion process";
        case 'prediction':
          return "High-accuracy predictions show successful training, with 98% denoising performance";
        case 'loss':
          return "Final loss value of 0.1 indicates successful convergence of the training process";
        case 'complete':
          return "LoRA training complete! Fine-tuned with minimal parameters while maintaining base model performance";
        default:
          return "LoRA adapter training complete with optimized parameters";
      }
    }
    
    // Standard explanations during normal flow
    switch(component) {
      case 'baseModel':
        return "Pre-trained language model with billions of parameters that remain frozen during training";
      case 'lora':
        return "Small, trainable adapter matrices (typically < 1% of model size) that modify the base model's behavior";
      case 'noise':
        if (elapsedTime < 4) return "Controlled noise is added to help the model learn denoising";
        return "Noise level decreases as training progresses";
      case 'prediction':
        if (elapsedTime < 6) return "Model predicts original input based on noisy data";
        return "Prediction accuracy improves over time";
      case 'loss':
        return "Measures difference between prediction and target; smaller is better";
      case 'matrix':
        return "LoRA uses low-rank matrices (A×B) to efficiently approximate weight changes";
      case 'complete':
        return "Training complete! All components working together with optimized parameters";
      default:
        return "";
    }
  };

  // Matrix animation for LoRA weight updates
  const getMatrixElements = () => {
    // Matrix A (smaller dimensions)
    const matrixAElements: React.ReactNode[] = [];
    const aRows = 4; // Reduce number of rows
    const aCols = 2;
    
    for (let i = 0; i < aRows; i++) {
      for (let j = 0; j < aCols; j++) {
        const value = Math.sin(elapsedTime * 2 + i + j) * 0.5 + 0.5;
        const intensity = Math.floor(value * 255).toString(16).padStart(2, '0');
        const color = `#${intensity}${intensity}ff`;
        
        matrixAElements.push(
          <rect
            key={`matrixA-${i}-${j}`}
            x={loraX + 15 + j * 15}
            y={loraY + 65 + i * 12} // Reduce spacing between matrix elements
            width={12}
            height={10} // Slightly smaller height
            fill={color}
            opacity={matrixUpdateVisible ? 0.9 : 0.4}
          />
        );
      }
    }
    
    // Matrix B (smaller dimensions)
    const matrixBElements: React.ReactNode[] = [];
    const bRows = 2;
    const bCols = 4; // Reduce number of columns
    
    for (let i = 0; i < bRows; i++) {
      for (let j = 0; j < bCols; j++) {
        const value = Math.cos(elapsedTime * 2.5 + i + j) * 0.5 + 0.5;
        const intensity = Math.floor(value * 255).toString(16).padStart(2, '0');
        const color = `#ff${intensity}${intensity}`;
        
        matrixBElements.push(
          <rect
            key={`matrixB-${i}-${j}`}
            x={loraX + 15 + j * 15}
            y={loraY + 125 + i * 12} // Moved up and reduced spacing
            width={12}
            height={10} // Slightly smaller height
            fill={color}
            opacity={matrixUpdateVisible ? 0.9 : 0.4}
          />
        );
      }
    }
    
    return { matrixAElements, matrixBElements };
  };

  // Show internal architecture elements for the base model
  const getModelInternalElements = () => {
    const internalElements: React.ReactNode[] = [];
    const layersCount = 5; // Reduce layer count
    
    // Add attention head representations
    for (let layer = 0; layer < layersCount; layer++) {
      const y = baseModelY + 50 + layer * (baseModelHeight - 70) / layersCount;
      
      // Add attention heads
      for (let head = 0; head < 4; head++) {
        const headX = baseModelX + 30 + head * ((baseModelWidth - 60) / 4);
        const pulse = Math.sin(elapsedTime * 3 + layer + head) * 0.5 + 0.5;
        
        internalElements.push(
          <circle
            key={`attention-${layer}-${head}`}
            cx={headX}
            cy={y + (baseModelHeight - 70) / layersCount / 2}
            r={5}
            fill="white"
            opacity={0.3 + pulse * 0.4}
          />
        );
        
        // Connection lines between attention heads
        if (head < 3) {
          internalElements.push(
            <line
              key={`connection-${layer}-${head}`}
              x1={headX + 5}
              y1={y + (baseModelHeight - 70) / layersCount / 2}
              x2={headX + ((baseModelWidth - 60) / 4) - 5}
              y2={y + (baseModelHeight - 70) / layersCount / 2}
              stroke="white"
              strokeWidth={1}
              opacity={0.2}
            />
          );
        }
        
        // Vertical connections between layers
        if (layer < layersCount - 1) {
          internalElements.push(
            <line
              key={`vertical-${layer}-${head}`}
              x1={headX}
              y1={y + (baseModelHeight - 70) / layersCount / 2 + 5}
              x2={headX}
              y2={y + (baseModelHeight - 70) / layersCount + 5}
              stroke="white" 
              strokeWidth={1}
              opacity={0.2}
              strokeDasharray="3,3"
            />
          );
        }
      }
      
      // Layer label
      internalElements.push(
        <text
          key={`layer-label-${layer}`}
          x={baseModelX + 15}
          y={y + (baseModelHeight - 70) / layersCount / 2 + 3}
          fontSize={7}
          fill="white"
          opacity={0.7}
        >
          {layer === 0 ? "Input" : layer === layersCount - 1 ? "Output" : `Layer ${layer}`}
        </text>
      );
    }
    
    return internalElements;
  };
  
  // Render component explanations if enabled
  const renderExplanations = () => {
    if (!showExplanations) return null;
    
    // Get the current focus area
    const focusArea = getCurrentFocusArea();
    // Get the explanation for the current focus
    let focusExplanation = "";
    
    // Special explanation for full visibility mode
    if (fullVisibilityMode) {
      focusExplanation = "LoRA training complete! Low-rank adaptation modules efficiently fine-tune the model with minimal parameters.";
    } else {
      switch(focusArea) {
        case 'baseModel':
          focusExplanation = getExplanationFor('baseModel');
          break;
        case 'dataFlow':
          focusExplanation = "Data flows through the frozen base model for initial processing";
          break;
        case 'noise':
          focusExplanation = getExplanationFor('noise');
          break;
        case 'prediction':
          focusExplanation = getExplanationFor('prediction');
          break;
        case 'loss':
          focusExplanation = getExplanationFor('loss');
          break;
        case 'lora':
          focusExplanation = getExplanationFor('lora');
          break;
        case 'feedback':
          focusExplanation = "Updated parameters flow back to influence the next training iteration";
          break;
        default:
          focusExplanation = getCurrentStageDescription();
      }
    }
    
    // Create a single explanation box at the top
    return (
      <g key="main-explanation" className="explanation-box">
        <rect
          x={innerWidth * 0.1}
          y={10}
          width={innerWidth * 0.8}
          height={40}
          rx={5}
          fill={fullVisibilityMode ? "rgba(30, 30, 60, 0.85)" : "rgba(0, 0, 0, 0.8)"}
          opacity={0.9}
          style={fullVisibilityMode ? { animation: 'pulse-glow 3s infinite ease-in-out' } : {}}
        />
        <text
          x={innerWidth * 0.5}
          y={30}
          fontSize={fullVisibilityMode ? 16 : 14}
          fill={fullVisibilityMode ? "#ffd700" : "white"}
          textAnchor="middle"
          fontWeight={fullVisibilityMode ? "bold" : "normal"}
          dominantBaseline="middle"
        >
          {focusExplanation}
        </text>
      </g>
    );
  };

  // Render technical explanation box at the bottom
  const renderTechnicalExplanation = () => {
    if (!showExplanations) return null;
    
    // Get the current focus area
    const focusArea = getCurrentFocusArea();
    // Get the technical explanation for the current focus
    let techExplanation = "";
    
    switch(focusArea) {
      case 'baseModel':
        techExplanation = "SDXL UNet2DConditionModel with text encoders and VAE frozen using requires_grad_(False)";
        break;
      case 'dataFlow':
        techExplanation = "VAE encodes images to latent space (latent_dist = vae.encode(pixel_values)) and CLIP tokenizers process text";
        break;
      case 'noise':
        techExplanation = "noise_scheduler.add_noise(latents, noise, timesteps) with randomly sampled diffusion timesteps";
        break;
      case 'prediction':
        techExplanation = "UNet forward pass with cross-attention to predict noise: unet(noisy_latents, timesteps, encoder_hidden_states)";
        break;
      case 'loss':
        techExplanation = "MSE loss between predicted and actual noise: F.mse_loss(model_pred, target, reduction='mean')";
        break;
      case 'lora':
        techExplanation = "LoRA with low rank matrices targeting attention, projection and convolution layers";
        break;
      case 'feedback':
        techExplanation = "Gradient accumulation with accelerator.backward(loss) and get_peft_model_state_dict() for checkpoints";
        break;
      default:
        techExplanation = "Deep learning techniques for efficient fine-tuning with minimal parameters";
    }
    
    // Position directly below the base model
    const techBoxY = baseModelY + baseModelHeight + 10;
    const techBoxWidth = baseModelWidth * 3; // Three times wider
    const techBoxX = (innerWidth - techBoxWidth) / 2; // Center horizontally
    
    // Create a technical explanation box at the bottom of the base model
    return (
      <g key="tech-explanation" className="tech-explanation-box">
        <rect
          x={techBoxX}
          y={techBoxY}
          width={techBoxWidth}
          height={40}
          rx={5}
          fill="rgba(40, 40, 80, 0.9)"
          opacity={0.9}
        />
        <text
          x={techBoxX + techBoxWidth / 2}
          y={techBoxY + 20}
          fontSize={10}
          fill="white"
          textAnchor="middle"
          dominantBaseline="middle"
          fontFamily="monospace"
        >
          {techExplanation}
        </text>
      </g>
    );
  };

  // Animation update logic for matrix elements and explanations
  useEffect(() => {
    setMatrixUpdateVisible(elapsedTime > 3 && lossValue < 0.8);
  }, [elapsedTime, lossValue]);

  // Start animation properly when component mounts
  useEffect(() => {
    // First ensure initial state is properly set
    setElapsedTime(0);
    setNoiseLevel(0);
    setPredictionAccuracy(0);
    setLossValue(1);
    setLoraActive(false);
    setFocusedElement(null);
    
    // Initialize data points
    const initialPoints: DataPoint[] = [];
    for (let i = 0; i < 25; i++) {
      initialPoints.push(createDataPoint(i));
    }
    setDataPoints(initialPoints);
    
    // Delay animation start to ensure DOM is ready and all state is applied
    const timer = setTimeout(() => {
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

  // Create a new data point with more randomization
  const createDataPoint = (id: number): DataPoint => {
    // Add some randomization to starting position to prevent clumping
    const startOffset = Math.random() * 15;
    return {
      id,
      x: margin.left - startOffset * 5, // Add randomization to x position
      y: innerHeight * 0.23 + (Math.random() * 6 - 3), // Add slight randomization to y position
      size: 4 + Math.random() * 6, // Slightly smaller base size
      opacity: 0.6 + Math.random() * 0.4,
      color: colors.dataFlow,
      speedFactor: 0.7 + Math.random() * 0.6 // More variance in speed
    };
  };

  // Helper function for linear interpolation
  const lerp = (a: number, b: number, t: number) => {
    return a + (b - a) * t;
  };

  // Define colors and theme
  const colors = {
    primary: '#3b82f6',       // Primary blue
    secondary: '#f43f5e',     // Accent pink
    highlight: '#10b981',     // Success green
    background: '#f1f5f9',    // Light background
    text: '#1e293b',          // Dark text
    lightText: '#64748b',     // Light text
    gradient1: '#4338ca',     // Gradient start (indigo)
    gradient2: '#3b82f6',     // Gradient end (blue)
    dataFlow: '#c026d3',      // Magenta for data flow
    noise: '#ea580c',         // Orange for noise
    lora: '#eab308',          // Yellow for LoRA
    prediction: '#059669',    // Teal for prediction
    explanation: 'rgba(17, 24, 39, 0.75)' // Semi-transparent background for explanations
  };

  // Margins for the visualization - increase to prevent cut-off
  const margin = { top: 30, right: 60, bottom: 20, left: 60 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  
  // Layout dimensions - adjust to fit within visible area
  const baseModelWidth = innerWidth * 0.25;
  const baseModelHeight = innerHeight * 0.55; // Reduce height to prevent overflow
  const baseModelX = innerWidth * 0.18;
  const baseModelY = innerHeight * 0.08;
  
  const loraWidth = innerWidth * 0.12;
  const loraHeight = innerHeight * 0.35;     // Reduce height to match base model
  const loraX = innerWidth * 0.65;          // Move left slightly to avoid right edge
  const loraY = innerHeight * 0.18;         // Move up to align with base model
  
  // Training flow paths
  const inputPath = [
    { x: 20, y: innerHeight * 0.23 },         // Move up to match base model
    { x: baseModelX - 20, y: innerHeight * 0.23 }
  ];
  
  const throughModelPath = [
    { x: baseModelX + baseModelWidth, y: innerHeight * 0.23 },           // Start exactly at model edge
    { x: baseModelX + baseModelWidth + 40, y: innerHeight * 0.23 },      // Horizontal segment
    { x: innerWidth * 0.55, y: innerHeight * 0.33 },                     // Move further right
    { x: innerWidth * 0.55, y: innerHeight * 0.43 },                     // Vertical segment to prediction start
  ];
  
  const predictionPath = [
    { x: innerWidth * 0.55, y: innerHeight * 0.43 },                     // Start matching the throughModelPath end
    { x: innerWidth * 0.48, y: innerHeight * 0.51 },                     // Curve toward loss
    { x: innerWidth * 0.42, y: innerHeight * 0.58 },                     // End exactly at loss circle
  ];
  
  const lossPath = [
    { x: innerWidth * 0.42, y: innerHeight * 0.58 },  // Start from loss circle center
    { x: innerWidth * 0.55, y: innerHeight * 0.58 },  // Horizontal control point at same height
    { x: loraX, y: loraY + loraHeight * 0.75 },       // Connect directly to LoRA module side
  ];
  
  const loraFeedbackPath = [
    { x: loraX + loraWidth * 0.5, y: loraY + loraHeight },
    { x: loraX + loraWidth * 0.5, y: loraY + loraHeight + 5 },
    { x: innerWidth * 0.55, y: innerHeight * 0.65 },
    { x: innerWidth * 0.35, y: innerHeight * 0.65 },
    { x: baseModelX - 15, y: innerHeight * 0.45 },
    { x: baseModelX - 15, y: innerHeight * 0.33 },
    { x: baseModelX, y: innerHeight * 0.33 },
  ];
  
  // New paths connecting to LoRA
  const noiseToLoraPath = [
    // Empty - noise doesn't directly train LoRA
  ];
  
  const predictionToLoraPath = [
    // Empty - prediction doesn't directly train LoRA
  ];

  return (
    <div className="training-process-visual">
      <FadeIn>
        <div className="visual-controls mb-4 flex items-center justify-between bg-slate-100 p-4 rounded-lg">
          <div>
            <h3 className="text-lg font-semibold text-slate-800">LoRA Training Flow</h3>
            <p className="text-sm text-slate-600">
              {getCurrentStageDescription()}
            </p>
          </div>
          <div className="flex gap-2">
            <button 
              className={`px-3 py-1.5 rounded-md ${isAnimating ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'}`}
              onClick={isAnimating ? stopAnimation : startAnimation}
            >
              {isAnimating ? 'Pause Flow' : 'Auto Play'}
            </button>
            
            <button 
              className="px-3 py-1.5 rounded-md bg-slate-300 text-slate-700"
              onClick={resetAnimation}
            >
              Reset
            </button>
            
            {/* Add button for final step without full effect */}
            <button 
              className="px-3 py-1.5 rounded-md bg-indigo-500 text-white"
              onClick={jumpToFinalStep}
              title="Skip to the final training step"
            >
              Final Step
            </button>
            
            {/* Full visibility mode button with more descriptive text */}
            <button 
              className={`px-3 py-1.5 rounded-md ${fullVisibilityMode ? 'bg-green-500 text-white' : 'bg-purple-500 text-white'}`}
              onClick={toggleFullVisibilityMode}
              title="Show all components with enhanced effects"
            >
              {fullVisibilityMode ? 'Normal Mode' : 'Full Animation'}
            </button>
            
            <div className="flex items-center gap-2 ml-2">
              <label className="text-sm text-slate-700">Speed:</label>
              <input 
                type="range" 
                min="0.5" 
                max="3" 
                step="0.1" 
                value={animationSpeed}
                onChange={(e) => setAnimationSpeed(Number(e.target.value))}
                className="w-20"
              />
              <span className="text-sm text-slate-700">{animationSpeed.toFixed(1)}x</span>
            </div>
            
            <button
              className={`px-3 py-1.5 rounded-md ${showExplanations ? 'bg-green-500 text-white' : 'bg-slate-300 text-slate-700'}`}
              onClick={() => setShowExplanations(!showExplanations)}
            >
              {showExplanations ? 'Hide Labels' : 'Show Labels'}
            </button>
          </div>
        </div>
        
        <div className="bg-white border rounded-lg shadow-sm overflow-hidden">
          <svg 
            width={width} 
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ maxHeight: "100%" }}
          >
            {/* Define gradients and markers */}
            <LinearGradient
              id="baseModelGradient"
              from={colors.gradient1}
              to={colors.gradient2}
              vertical={true}
            />
            
            <LinearGradient
              id="loraGradient"
              from={colors.lora}
              to="#e67e22"
              vertical={true}
            />
            
            <RadialGradient
              id="glowGradient"
              from={colors.lora}
              to="transparent"
              fromOffset={0}
              toOffset={1}
              r="50%"
              cx="50%"
              cy="50%"
            />
            
            <MarkerArrow
              id="arrow"
              fill={colors.primary}
              refX={2}
              size={6}
            />
            
            <MarkerCross
              id="loss"
              fill={colors.secondary}
              size={6}
              strokeWidth={2}
            />
            
            <Group top={margin.top} left={margin.left}>
              {/* Background */}
              <rect
                width={innerWidth}
                height={innerHeight}
                fill={colors.background}
                rx={10}
                opacity={0.3}
              />
              
              {/* Flow paths */}
              <Line
                from={{ x: inputPath[0].x, y: inputPath[0].y }}
                to={{ x: inputPath[1].x, y: inputPath[1].y }}
                stroke={colors.primary}
                strokeWidth={2}
                markerEnd="url(#arrow)"
                opacity={1}
                style={getHighlightStyle('dataFlow')}
                onClick={() => handleElementClick('dataFlow')}
                cursor="pointer"
              />
              
              <LinePath
                data={throughModelPath}
                x={d => d.x}
                y={d => d.y}
                stroke={colors.primary}
                strokeWidth={2}
                curve={curveBasis}
                markerEnd="url(#arrow)"
                opacity={getElementOpacity('dataFlow')}
                style={getHighlightStyle('dataFlow')}
                onClick={() => handleElementClick('dataFlow')}
                cursor="pointer"
              />
              
              <LinePath
                data={predictionPath}
                x={d => d.x}
                y={d => d.y}
                stroke={colors.prediction}
                strokeWidth={2}
                curve={curveBasis}
                opacity={getElementOpacity('prediction')}
                style={getHighlightStyle('prediction')}
                onClick={() => handleElementClick('prediction')}
                cursor="pointer"
              />
              
              <LinePath
                data={lossPath}
                x={d => d.x}
                y={d => d.y}
                stroke={colors.secondary}
                strokeWidth={2}
                curve={curveBasis}
                markerEnd="url(#loss)"
                opacity={getElementOpacity('loss')}
                style={getHighlightStyle('loss')}
                onClick={() => handleElementClick('loss')}
                cursor="pointer"
              />
              
              <LinePath
                data={loraFeedbackPath}
                x={d => d.x}
                y={d => d.y}
                stroke={colors.lora}
                strokeWidth={2}
                strokeDasharray="4,4"
                curve={curveBasis}
                markerEnd="url(#arrow)"
                opacity={getElementOpacity('feedback')}
                style={getHighlightStyle('feedback')}
                onClick={() => handleElementClick('feedback')}
                cursor="pointer"
              />

              {/* Noise visualization */}
              <g 
                className="noise-visualization" 
                style={{ 
                  ...getHighlightStyle('noise'),
                  opacity: Math.min(1, noiseLevel * 1.5) * getElementOpacity('noise'),
                  cursor: "pointer"
                }}
                onClick={() => handleElementClick('noise')}
              >
                <circle
                  cx={innerWidth * 0.48}
                  cy={innerHeight * 0.33}
                  r={20}
                  fill={colors.noise}
                  opacity={0.2}
                />
                
                {Array.from({ length: 10 }).map((_, i) => {
                  const angle = (i / 10) * Math.PI * 2;
                  const dist = 15 + Math.sin(i * 5) * 5;
                  
                  return (
                    <circle
                      key={`noise-${i}`}
                      cx={innerWidth * 0.48 + Math.cos(angle) * dist}
                      cy={innerHeight * 0.33 + Math.sin(angle) * dist}
                      r={2 + Math.random() * 2}
                      fill={colors.noise}
                      opacity={0.3 + Math.sin(i + elapsedTime * 5) * 0.3 * noiseLevel}
                    />
                  );
                })}
                
                <Text
                  x={innerWidth * 0.48}
                  y={innerHeight * 0.33 - 30}
                  textAnchor="middle"
                  verticalAnchor="middle"
                  fill={colors.noise}
                  fontSize={12}
                  fontWeight={500}
                  opacity={noiseLevel * 0.8}
                >
                  Noise Addition
                </Text>
              </g>
              
              {/* Prediction visualization */}
              <g 
                className="prediction-visualization" 
                style={{ 
                  ...getHighlightStyle('prediction'),
                  opacity: Math.min(1, predictionAccuracy * 1.5) * getElementOpacity('prediction'),
                  cursor: "pointer"
                }}
                onClick={() => handleElementClick('prediction')}
              >
                <circle
                  cx={innerWidth * 0.5}
                  cy={innerHeight * 0.51}
                  r={20}
                  fill={colors.prediction}
                  opacity={0.2}
                />
                
                <path
                  d={`M ${innerWidth * 0.5 - 15} ${innerHeight * 0.51} 
                      L ${innerWidth * 0.5 - 5} ${innerHeight * 0.51 + 10} 
                      L ${innerWidth * 0.5 + 15} ${innerHeight * 0.51 - 10}`}
                  stroke={colors.prediction}
                  strokeWidth={2 + predictionAccuracy * 2}
                  fill="none"
                  opacity={0.4 + predictionAccuracy * 0.6}
                />
                
                <Text
                  x={innerWidth * 0.5}
                  y={innerHeight * 0.51 + 30}
                  textAnchor="middle"
                  verticalAnchor="middle"
                  fill={colors.prediction}
                  fontSize={12}
                  fontWeight={500}
                  opacity={predictionAccuracy * 0.8}
                >
                  Prediction
                </Text>
              </g>
              
              {/* Loss visualization */}
              <g 
                className="loss-visualization" 
                style={{ 
                  ...getHighlightStyle('loss'),
                  opacity: Math.min(1, lossValue * 1.2) * getElementOpacity('loss'),
                  cursor: "pointer"
                }}
                onClick={() => handleElementClick('loss')}
              >
                <circle
                  cx={innerWidth * 0.42}
                  cy={innerHeight * 0.58}
                  r={20}
                  fill={colors.secondary}
                  opacity={0.2}
                />
                
                <text
                  x={innerWidth * 0.42}
                  y={innerHeight * 0.58 + 5}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={colors.secondary}
                  fontSize={14 - lossValue * 4}
                  fontWeight="600"
                >
                  {lossValue.toFixed(2)}
                </text>
                
                <Text
                  x={innerWidth * 0.42}
                  y={innerHeight * 0.58 + 30}
                  textAnchor="middle"
                  verticalAnchor="middle"
                  fill={colors.secondary}
                  fontSize={12}
                  fontWeight={500}
                >
                  Loss
                </Text>
              </g>
              
              {/* Data particles */}
              {dataPoints.map(point => (
                <circle
                  key={`data-${point.id}`}
                  cx={point.x}
                  cy={point.y}
                  r={point.size}
                  fill={point.color}
                  opacity={point.opacity * getElementOpacity(point.color === colors.dataFlow ? 'dataFlow' : 
                                         point.color === colors.noise ? 'noise' : 
                                         point.color === colors.prediction ? 'prediction' : 
                                         point.color === colors.secondary ? 'loss' : 
                                         point.color === colors.lora ? 'lora' : 'dataFlow')}
                />
              ))}

              {/* Base model */}
              <g 
                className="base-model" 
                style={{ 
                  ...getHighlightStyle('baseModel'),
                  opacity: baseModelSpring.opacity.get() * getElementOpacity('baseModel'), 
                  transform: `scale(${baseModelSpring.scale.get()})`,
                  transformOrigin: `${baseModelX + baseModelWidth/2}px ${baseModelY + baseModelHeight/2}px`,
                  cursor: "pointer"
                }}
                onClick={() => handleElementClick('baseModel')}
              >
                <rect
                  x={baseModelX}
                  y={baseModelY}
                  width={baseModelWidth}
                  height={baseModelHeight}
                  rx={15}
                  fill="url(#baseModelGradient)"
                  className="shadow-lg"
                />
                
                <Text
                  x={baseModelX + baseModelWidth / 2}
                  y={baseModelY + 30}
                  textAnchor="middle"
                  verticalAnchor="middle"
                  fill="white"
                  fontWeight={600}
                  fontSize={16}
                >
                  Frozen Base Model
                </Text>
                
                {/* Internal model architecture */}
                {getModelInternalElements()}
                
                {/* Lock icon to indicate frozen */}
                <circle
                  cx={baseModelX + baseModelWidth - 20}
                  cy={baseModelY + 20}
                  r={10}
                  fill="white"
                  opacity={0.8}
                />
                <path
                  d={`M ${baseModelX + baseModelWidth - 25} ${baseModelY + 19}
                      a 5 5 0 0 1 10 0
                      v 3
                      h 2
                      v 7
                      h -14
                      v -7
                      h 2
                      v -3`}
                  fill={colors.gradient1}
                  opacity={0.8}
                />
              </g>
              
              {/* LoRA module */}
              <g 
                className="lora-module" 
                style={{ 
                  ...getHighlightStyle('lora'),
                  opacity: loraSpring.opacity.get() * getElementOpacity('lora'), 
                  transform: `scale(${loraSpring.scale.get()})`,
                  transformOrigin: `${loraX + loraWidth/2}px ${loraY + loraHeight/2}px`,
                  cursor: "pointer"
                }}
                onClick={() => handleElementClick('lora')}
              >
                {/* Glow effect */}
                <circle
                  cx={loraX + loraWidth / 2}
                  cy={loraY + loraHeight / 2}
                  r={loraWidth}
                  fill="url(#glowGradient)"
                  opacity={loraSpring.glow.get() * 0.03}
                />
                
                <rect
                  x={loraX}
                  y={loraY}
                  width={loraWidth}
                  height={loraHeight}
                  rx={10}
                  fill="url(#loraGradient)"
                />
                
                <Text
                  x={loraX + loraWidth / 2}
                  y={loraY + 30}
                  textAnchor="middle"
                  verticalAnchor="middle"
                  fill="white"
                  fontWeight={600}
                  fontSize={16}
                >
                  LoRA Adapters
                </Text>
                
                {/* Matrix representation */}
                <text
                  x={loraX + loraWidth / 2}
                  y={loraY + 50}
                  textAnchor="middle"
                  fill="white"
                  fontSize={12}
                  fontWeight={500}
                  opacity={matrixUpdateVisible ? 0.9 : 0.6}
                >
                  Matrix A
                </text>
                
                {getMatrixElements().matrixAElements}
                
                <text
                  x={loraX + loraWidth / 2}
                  y={loraY + 115}
                  textAnchor="middle"
                  fill="white"
                  fontSize={12}
                  fontWeight={500}
                  opacity={matrixUpdateVisible ? 0.9 : 0.6}
                >
                  Matrix B
                </text>
                
                {getMatrixElements().matrixBElements}
                
                {/* Matrix multiplication symbol */}
                {matrixUpdateVisible && (
                  <text
                    x={loraX + loraWidth / 2}
                    y={loraY + 160}
                    textAnchor="middle"
                    fill="white"
                    fontSize={16}
                    fontWeight={700}
                  >
                    = ΔW
                  </text>
                )}
              </g>
              
              {/* Explanations overlay */}
              {renderExplanations()}
              
              {/* Technical explanation at bottom */}
              {renderTechnicalExplanation()}
            </Group>
          </svg>
        </div>
        
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">Key LoRA Training Benefits</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-50 border p-4 rounded-lg">
              <h4 className="font-medium text-slate-800">Efficiency</h4>
              <p className="text-sm text-slate-600">Trains 99% fewer parameters than full fine-tuning</p>
            </div>
            <div className="bg-slate-50 border p-4 rounded-lg">
              <h4 className="font-medium text-slate-800">Quality</h4>
              <p className="text-sm text-slate-600">Produces results comparable to full fine-tuning</p>
            </div>
            <div className="bg-slate-50 border p-4 rounded-lg">
              <h4 className="font-medium text-slate-800">Speed</h4>
              <p className="text-sm text-slate-600">Trains in minutes instead of hours or days</p>
            </div>
            <div className="bg-slate-50 border p-4 rounded-lg">
              <h4 className="font-medium text-slate-800">Portability</h4>
              <p className="text-sm text-slate-600">Creates tiny files (MB vs GB) that are easy to share</p>
            </div>
          </div>
        </div>
      </FadeIn>
    </div>
  )
};

export default TrainingProcessVisual;