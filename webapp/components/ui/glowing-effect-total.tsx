"use client";

import { memo, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface GlowingEffectProps {
  blur?: number;
  inactiveZone?: number;
  proximity?: number;
  spread?: number;
  variant?: "default" | "white";
  glow?: boolean;
  className?: string;
  disabled?: boolean;
  movementDuration?: number;
  borderWidth?: number;
}
const GlowingEffect = memo(
  ({
    blur = 0,
    spread = 20,
    variant = "default",
    glow = true,
    className,
    borderWidth = 1,
    disabled = false,
  }: GlowingEffectProps) => {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      if (disabled || !containerRef.current) return;

      const element = containerRef.current;
      element.style.setProperty("--active", "1");

      let currentAngle =
        parseFloat(element.style.getPropertyValue("--start")) || 0;
      let animationId: number;

      const animateRotation = () => {
        currentAngle = (currentAngle + 1) % 360;
        element.style.setProperty("--start", String(currentAngle));
        animationId = requestAnimationFrame(animateRotation);
      };

      animateRotation();

      return () => {
        if (animationId) {
          cancelAnimationFrame(animationId);
        }
      };
    }, [disabled, glow]);

    return (
      <>
        <div
          className={cn(
            "pointer-events-none absolute -inset-px hidden rounded-[inherit] border opacity-0 transition-opacity",
            glow && "opacity-100",
            variant === "white" && "border-white",
            disabled && "!block"
          )}
        />
        <div
          ref={containerRef}
          style={
            {
              "--blur": `${blur}px`,
              "--spread": spread,
              "--start": "0",
              "--active": disabled ? "0" : "1",
              "--glowingeffect-border-width": `${borderWidth}px`,
              "--repeating-conic-gradient-times": "5",
              "--gradient":
                variant === "white"
                  ? `repeating-conic-gradient(
                  from 236.84deg at 50% 50%,
                  var(--black),
                  var(--black) calc(25% / var(--repeating-conic-gradient-times))
                )`
                  : `radial-gradient(circle, #22c55e 10%, #22c55e00 20%),
                radial-gradient(circle at 40% 40%, #16a34a 5%, #16a34a00 15%),
                radial-gradient(circle at 60% 60%, #15803d 10%, #15803d00 20%), 
                radial-gradient(circle at 40% 60%, #166534 10%, #16653400 20%),
                repeating-conic-gradient(
                  from calc(var(--start) * 1deg) at 50% 50%,
                  #22c55e 0%,
                  #16a34a calc(25% / var(--repeating-conic-gradient-times)),
                  #15803d calc(50% / var(--repeating-conic-gradient-times)), 
                  #166534 calc(75% / var(--repeating-conic-gradient-times)),
                  #22c55e calc(100% / var(--repeating-conic-gradient-times))
                )`,
            } as React.CSSProperties
          }
          className={cn(
            "pointer-events-none absolute inset-0 rounded-[inherit] opacity-0 transition-opacity",
            glow && "opacity-100",
            blur > 0 && "blur-[var(--blur)] ",
            className,
            disabled && "!hidden"
          )}
        >
          <div
            className={cn(
              "glow",
              "rounded-[inherit]",
              "after:content-[''] after:rounded-[inherit] after:absolute after:inset-[calc(-1*var(--glowingeffect-border-width))]",
              "after:[border:var(--glowingeffect-border-width)_dashed_transparent]",
              "after:[background:var(--gradient)] after:[background-attachment:fixed]",
              "after:opacity-[var(--active)] after:transition-opacity after:duration-300",
              "after:[mask-clip:padding-box,border-box]",
              "after:[mask-composite:intersect]",
              "after:[mask-image:linear-gradient(#0000,#0000),conic-gradient(from_calc((var(--start)-var(--spread))*1deg),#00000000_0deg,#fff,#00000000_calc(var(--spread)*2deg))]"
            )}
          />
        </div>
      </>
    );
  }
);

GlowingEffect.displayName = "GlowingEffect";

export { GlowingEffect };