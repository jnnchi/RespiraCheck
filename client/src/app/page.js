import Image from "next/image";
import { Typography, Box } from "@mui/material";

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
        <Box sx={{width: '50%'}}>
            <UploadButton></UploadButton>
            <Typography>Upload Audio</Typography>
        </Box>
        <Box sx={{width: '50%'}}>
            <Steps></Steps>
        </Box>
    </ThemeProvider>
  );
}
