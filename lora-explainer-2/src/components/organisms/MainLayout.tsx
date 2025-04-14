import * as React from "react"
import { cn } from "@/lib/utils"
import { Button } from "../atoms/buttons/button"
import { Input } from "../atoms/inputs/input"
import { Avatar, AvatarFallback, AvatarImage } from "../atoms/display/avatar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./navigation/dropdown-menu"
import { Bell, Menu, Search } from "lucide-react"

interface MainLayoutProps extends React.HTMLAttributes<HTMLDivElement> {
  user?: {
    name: string
    email: string
    image?: string
  }
  children: React.ReactNode
}

export function MainLayout({
  user,
  children,
  className,
  ...props
}: MainLayoutProps) {
  const [mobileNav, setMobileNav] = React.useState(false)

  return (
    <div className="flex min-h-screen flex-col">
      <header className="bg-background border-border flex h-14 items-center gap-4 border-b px-4 md:px-6">
        <div className="flex items-center gap-3">
          <svg
            width="32"
            height="32"
            viewBox="0 0 32 32"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="text-primary"
          >
            <rect
              x="4"
              y="4"
              width="24"
              height="24"
              rx="6"
              className="fill-current"
            />
            <path
              d="M16 10L20 14L16 18M12 14H20"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-lg font-semibold">Your App</span>
        </div>

        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={() => setMobileNav(true)}
        >
          <Menu className="size-5" />
          <span className="sr-only">Toggle menu</span>
        </Button>

        <div className="flex-1 md:flex-initial">
          <div className="relative">
            <Search className="text-muted-foreground absolute left-2.5 top-2.5 size-4" />
            <Input
              type="search"
              placeholder="Search..."
              className="bg-muted w-full pl-8 md:w-[200px] lg:w-[300px]"
            />
          </div>
        </div>

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

      <main
        className={cn("flex-1 overflow-auto", className)}
        {...props}
      >
        {children}
      </main>
    </div>
  )
} 