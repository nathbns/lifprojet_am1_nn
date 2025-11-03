import type { ReactNode } from 'react';

export default function Layout({ children }: { children: ReactNode }) {
  return <div className="max-w-6xl mx-auto px-4 sm:px-6 md:px-8 py-6 sm:py-8">{children}</div>;
}


