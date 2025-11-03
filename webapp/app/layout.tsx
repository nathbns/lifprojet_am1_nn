import type { Metadata } from "next";
import { Pixelify_Sans } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { Sidebar } from "@/components/sidebar";

const pixelify = Pixelify_Sans({
  variable: "--font-pixelify-sans",
  subsets: ["latin"],
  weight: ["400", "700"],
});

export const metadata: Metadata = {
  title: "LIF Project - Test de Modèles IA",
  description: "Interface de test pour les modèles d'intelligence artificielle",
  icons: {
    icon: "/icon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr" suppressHydrationWarning>
      <body className={`${pixelify.variable} ${pixelify.variable} ${pixelify.className} antialiased`}>
              <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
              <div className="relative min-h-screen flex">
                <Sidebar />
                <main className="flex-1 md:ml-16">{children}</main>
              </div>
              </ThemeProvider>
      </body>
    </html>
  );
}
