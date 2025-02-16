import { Box, Typography, Stack } from "@mui/material";

import Navbar from '../components/navbar';
import UploadButton from '../components/upload-button';
import Steps from '../components/steps';

export default function Action() {
    return (
      <ThemeProvider theme={theme}>
        <Navbar></Navbar>
        <Stack direction ="row" alignItems="center" sx={{ width: "90%" }} >
            <Box sx={{width: '30%', display: 'flex', alignItems: 'center'}}>
                <UploadButton></UploadButton>
                <Typography>Upload Audio</Typography>
            </Box>
            <Steps></Steps>

        </Stack>
        

      </ThemeProvider>
    );
  }