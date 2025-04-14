import * as React from "react"
import { cn } from "@/lib/utils"

interface StepIndicatorProps extends React.HTMLAttributes<HTMLDivElement> {
  steps: number
  currentStep: number
}

export function StepIndicator({
  steps,
  currentStep,
  className,
  ...props
}: StepIndicatorProps) {
  return (
    <div
      className={cn("flex items-center justify-center gap-2", className)}
      {...props}
    >
      {Array.from({ length: steps }).map((_, index) => (
        <div
          key={index}
          className={cn(
            "size-2 rounded-full transition-all",
            index === currentStep
              ? "bg-primary w-4"
              : index < currentStep
              ? "bg-primary"
              : "bg-muted"
          )}
        />
      ))}
    </div>
  )
} 