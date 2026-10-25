// Small helper used by every shadcn/ui component in components/ui/.
//
// cn() joins CSS class names. clsx drops empty or false values (so a class can
// be added only when a condition holds), and tailwind-merge removes Tailwind
// classes that a later class overrides (e.g. "p-2 p-4" becomes "p-4").
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Joins class names, skipping empty ones and resolving Tailwind conflicts. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
