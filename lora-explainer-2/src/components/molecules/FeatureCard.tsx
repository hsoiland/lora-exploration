import React from "react";
import { 
  Card, 
  CardContent,
  CardTitle, 
  CardDescription 
} from "@/components/molecules/display/card";
import { LucideIcon } from "lucide-react";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ 
  title, 
  description, 
  icon: Icon 
}) => {
  return (
    <Card className="bg-card/50 hover:bg-card/70 transition-colors">
      <CardContent>
        <CardTitle className="mb-4">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
        <div className="mt-6 flex justify-end">
          <Icon className="h-5 w-5 text-muted-foreground" />
        </div>
      </CardContent>
    </Card>
  );
};

export default FeatureCard; 