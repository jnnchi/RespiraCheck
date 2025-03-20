import { League_Spartan } from "next/font/google";

// If loading a variable font, you don't need to specify the font weight
const spartan = Inter({ subsets: ["latin"] });

export default function MyApp({ Component, pageProps }) {
  return <main className={spartan.className}></main>;
}
