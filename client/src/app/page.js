"use client";

import Image from "next/image";
import { Typography, Box, Stack } from "@mui/material";

import { ThemeProvider } from "@mui/material/styles";
import theme from "./theme/theme";
import "./globals.css";

import Steps from './components/steps';
import Navbar from "./components/navbar";

export default function Landing() {
  return (
    <ThemeProvider theme={theme}>
      <Navbar></Navbar>
      {/* <Router>
        <Navbar></Navbar>
        <Routes>
          <Route path="/" element={<Home />}></Route>
          <Route path="/info" element={<Info />}></Route>
          <Route path="/action" element={<Action />}></Route>
          <Route path="/results" element={<Results />}></Route>
        </Routes>
        
      </Router> */}
{/*       
        <Stack width= "100%" direction ="row" alignItems="center" sx={{ paddingTop: "60px", justifyContent: "center" }} >
            <Box sx={{gap : 7,width: "30%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"}}>
                <UploadButton></UploadButton>
                
                <Typography sx={{fontSize: "25px"}}>Upload Audio</Typography>
            </Box>
            <Steps></Steps>

        </Stack> */}
        
    </ThemeProvider>
  );
}
