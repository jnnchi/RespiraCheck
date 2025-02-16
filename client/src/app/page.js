import Image from "next/image";
import { Typography, Box, Stack } from "@mui/material";

import { ThemeProvider } from "@mui/material/styles";
import theme from "./theme/theme";
import "./globals.css";

import UploadButton from './components/upload-button';
import Steps from './components/steps';
import Navbar from "./components/navbar";

export default function Home() {
  return (
    <ThemeProvider theme={theme}>
      <Navbar></Navbar>
        <Stack width= "100%" direction ="row" alignItems="center" sx={{ justifyContent: "center" }} >
            <Box sx={{width: "30%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"}}>
                <UploadButton></UploadButton>
                <Typography sx={{fontSize: "30px"}}>Upload Audio</Typography>
            </Box>
            <Steps></Steps>

        </Stack>
    </ThemeProvider>
  );
}
