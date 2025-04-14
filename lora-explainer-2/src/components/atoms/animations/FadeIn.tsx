import React, { useEffect, useState } from 'react'

interface FadeInProps {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
}

/**
 * FadeIn animation component
 * 
 * Animates children fading in with configurable delay and duration
 */
export function FadeIn({ 
  children, 
  delay = 0, 
  duration = 1000,
  className 
}: FadeInProps) {
  const [isVisible, setIsVisible] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, delay);
    
    return () => clearTimeout(timer);
  }, [delay]);
  
  const style = {
    opacity: isVisible ? 1 : 0,
    transition: `opacity ${duration}ms ease-in-out`,
  };

  return (
    <div style={style} className={className}>
      {children}
    </div>
  );
} 