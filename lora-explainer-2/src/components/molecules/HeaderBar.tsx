import * as React from "react"
import { Menu, Search, Bell } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/atoms/buttons"
import { Input } from "@/components/atoms/inputs"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/atoms/display"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/atoms/dropdown-menu"

interface HeaderBarProps extends React.HTMLAttributes<HTMLDivElement> {
  user?: {
    name: string
    email: string
    image?: string
  }
  onMenuClick?: () => void
  showSearch?: boolean
  onSearch?: (value: string) => void
}

export function HeaderBar({
  user,
  onMenuClick,
  showSearch = true,
  onSearch,
  className,
  ...props
}: HeaderBarProps) {
  return (
    <header
      className={cn(
        "bg-background border-border flex h-14 items-center gap-4 border-b px-4 md:px-6",
        className
      )}
      {...props}
    >
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden"
        onClick={onMenuClick}
      >
        <Menu className="size-5" />
        <span className="sr-only">Toggle menu</span>
      </Button>

      {showSearch && (
        <div className="flex-1 md:flex-initial">
          <div className="relative">
            <Search className="text-muted-foreground absolute left-2.5 top-2.5 size-4" />
            <Input
              type="search"
              placeholder="Search..."
              className="bg-muted w-full pl-8 md:w-[200px] lg:w-[300px]"
              onChange={(e) => onSearch?.(e.target.value)}
            />
          </div>
        </div>
      )}

      <div className="flex flex-1 items-center justify-end gap-4">
        <Button variant="ghost" size="icon">
          <Bell className="size-5" />
          <span className="sr-only">Notifications</span>
        </Button>

        {user && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="relative size-8 rounded-full"
              >
                <Avatar className="size-8">
                  <AvatarImage
                    src={user.image}
                    alt={user.name}
                  />
                  <AvatarFallback>
                    {user.name
                      .split(" ")
                      .map((n) => n[0])
                      .join("")
                      .toUpperCase()}
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium">{user.name}</p>
                  <p className="text-muted-foreground text-xs">{user.email}</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Profile</DropdownMenuItem>
              <DropdownMenuItem>Settings</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Log out</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>
    </header>
  )
} 