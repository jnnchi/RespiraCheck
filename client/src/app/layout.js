import { Geist, Geist_Mono } from "next/font/google";

import { League_Spartan } from "next/font/google";

import "./globals.css";

const spartan = League_Spartan({
  subsets: ["latin"],
  weight: ["100", "200", "300", "400", "500", "600", "700"],
});

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "RespiraCheck",
  description: "RespiraCheck - Your AI-powered COVID diagnosis tool.",
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${spartan.variable} ${spartan.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
