import * as React from "react"
import { Home, LayoutDashboard, Settings, Users } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/atoms/buttons"
import { ScrollArea } from "@/components/organisms/data/scroll-area"
import { Separator } from "@/components/atoms/display"

interface NavigationItem {
  title: string
  href: string
  icon?: React.ComponentType<{ className?: string }>
}

interface NavigationMenuProps extends React.HTMLAttributes<HTMLDivElement> {
  items?: NavigationItem[]
}

const defaultItems: NavigationItem[] = [
  {
    title: "Home",
    href: "/",
    icon: Home,
  },
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "Users",
    href: "/users",
    icon: Users,
  },
  {
    title: "Settings",
    href: "/settings",
    icon: Settings,
  },
]

export function NavigationMenu({
  items = defaultItems,
  className,
  ...props
}: NavigationMenuProps) {
  const [active, setActive] = React.useState(items[0]?.href)

  return (
    <div
      className={cn(
        "bg-background border-border flex h-full w-full flex-col border-r",
        className
      )}
      {...props}
    >
      <div className="flex h-14 items-center border-b px-6">
        <span className="font-semibold">Navigation</span>
      </div>
      <ScrollArea className="flex-1 px-2 py-2">
        <div className="space-y-1">
          {items.map((item) => {
            const Icon = item.icon
            return (
              <Button
                key={item.href}
                variant={active === item.href ? "secondary" : "ghost"}
                className="w-full justify-start"
                onClick={() => setActive(item.href)}
              >
                {Icon && <Icon className="mr-2 size-4" />}
                {item.title}
              </Button>
            )
          })}
        </div>
      </ScrollArea>
      <Separator />
      <div className="p-4">
        <p className="text-muted-foreground text-xs">
          Version 1.0.0
        </p>
      </div>
    </div>
  )
} 