import { Box, Typography, Stack } from "@mui/material";

import Navbar from '../components/navbar';
import UploadButton from '../components/upload-button';
import Steps from '../components/steps';

import { ThemeProvider } from "@mui/material/styles";
import theme from "../theme/theme";

export default function Action() {
    return (
      <ThemeProvider theme={theme}>
        {/* <Navbar></Navbar> */}
        <Stack width= "100%" direction ="row" alignItems="center" sx={{ justifyContent: "center" }} >
            <Box sx={{width: "30%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center"}}>
                <UploadButton></UploadButton>
                <Typography>Upload Audio</Typography>
            </Box>
            <Steps></Steps>

        </Stack>
        

      </ThemeProvider>
    );
  }