import Image from "next/image";
import Navbar from "./components/navbar";
import { Typography } from "@mui/material";

import { ThemeProvider } from "@mui/material/styles";
import theme from "./theme/theme";
import "./globals.css";

export default function Home() {
  return (
    <ThemeProvider theme={theme}>
      <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
          <Navbar></Navbar>
      </div>
    </ThemeProvider>
  );
}
