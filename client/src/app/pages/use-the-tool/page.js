import { ThemeProvider } from "@mui/material/styles";
import { Stack } from "@mui/material"
import theme from "../../theme/theme";
import UploadAudio from "@/app/components/upload-audio";
import RecordAudio from "@/app/components/record-audio";
import SubmitAudioHeading from "@/app/components/loading/submit-audio-heading";
import Navbar from "@/app/components/navbar";
import Link from "next/link";

export default function Action() {
  return (
      <ThemeProvider theme={theme}>
        <Navbar></Navbar>
    
        
        <Stack width= "100%" direction ="column" alignItems="center" spacing={5} sx={{ justifyContent: "center", mt: 6 }} >
          <SubmitAudioHeading/>

          <Stack width= "100%" direction ="row" alignItems="center" spacing={22} sx={{ justifyContent: "center" }} >
            <UploadAudio></UploadAudio>
            <RecordAudio></RecordAudio>
          </Stack>

        </Stack>

      <Link href="/pages/results"> 
        <span style={{ textDecoration: "underline"}}>Results</span>
      </Link>
      
    </ThemeProvider>
  );
}