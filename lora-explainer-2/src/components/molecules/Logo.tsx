import React from "react";

interface LogoProps {
  size?: "sm" | "md" | "lg";
}

const Logo: React.FC<LogoProps> = ({ size = "md" }) => {
  const sizeClasses = {
    sm: {
      outer: "size-16",
      middle: "size-10",
      inner: "size-6",
    },
    md: {
      outer: "size-24",
      middle: "size-16",
      inner: "size-12",
    },
    lg: {
      outer: "size-32",
      middle: "size-24",
      inner: "size-16",
    },
  };

  const classes = sizeClasses[size];

  return (
    <div className="relative">
      <div className={`${classes.outer} rounded-full bg-gradient-to-r from-amber-500/50 to-purple-500/50 flex items-center justify-center`}>
        <div className={`${classes.middle} rounded-full bg-background flex items-center justify-center`}>
          <div className={`${classes.inner} rounded-full bg-gradient-to-r from-amber-500 to-purple-500`}></div>
        </div>
      </div>
    </div>
  );
};

export default Logo; 