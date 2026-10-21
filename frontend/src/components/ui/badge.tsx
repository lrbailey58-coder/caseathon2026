import { cva, type VariantProps } from 'class-variance-authority'
import * as React from 'react'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-mono font-semibold uppercase tracking-wider transition-colors',
  {
    variants: {
      variant: {
        default: 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300',
        emerald: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
        amber: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
        rose: 'border-rose-500/40 bg-rose-500/10 text-rose-300',
        slate: 'border-slate-700 bg-slate-800/80 text-slate-300',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge }
