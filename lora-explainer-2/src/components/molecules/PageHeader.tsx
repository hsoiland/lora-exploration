import React from "react";

interface PageHeaderProps {
  username?: string;
}

const PageHeader: React.FC<PageHeaderProps> = ({ username = "M Anderson" }) => {
  return (
    <header className="flex justify-between items-center p-4 md:p-6">
      <div className="flex items-center gap-2">
        <span className="text-yellow-500 text-lg md:text-xl font-bold">
          🔍 ThreatCanary
        </span>
      </div>
      <div className="flex items-center">
        <span className="text-sm text-muted-foreground">{username}</span>
        <div className="ml-2 inline-block h-8 w-8 rounded-full bg-slate-600"></div>
      </div>
    </header>
  );
};

export default PageHeader; 