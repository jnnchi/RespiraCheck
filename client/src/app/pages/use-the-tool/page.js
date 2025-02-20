import { Box, Typography, Stack } from "@mui/material";

import UploadButton from '../../components/upload-button';
import Steps from '../../components/steps';
import Navbar from '../../components/navbar';

import { ThemeProvider } from "@mui/material/styles";
import theme from "../../theme/theme";
import UploadAudio from "@/app/components/upload-audio";
import RecordAudio from "@/app/components/record-audio";
import SubmitAudioHeading from "@/app/components/submit-audio-heading";

export default function Action() {
    return (
      <ThemeProvider theme={theme}>
        <Navbar></Navbar>
        {/* <Stack width= "100%" direction ="row" alignItems="center" sx={{ justifyContent: "center" }} >
            <Box sx={{width: "30%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"}}>
                <UploadButton></UploadButton>
                <Typography>Upload Audio</Typography>
            </Box>
            <Steps></Steps>

        </Stack> */}
      
      <Stack width= "100%" direction ="column" alignItems="center" spacing={10} sx={{ justifyContent: "center", mt: 6 }} >
        <SubmitAudioHeading></SubmitAudioHeading>

        <Stack width= "100%" direction ="row" alignItems="center" spacing={22} sx={{ justifyContent: "center" }} >
          <UploadAudio></UploadAudio>
          <RecordAudio></RecordAudio>
        </Stack>

      </Stack>
      
      

      
    </ThemeProvider>
  );
}