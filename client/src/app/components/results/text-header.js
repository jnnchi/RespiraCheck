import { Box, Typography } from "@mui/material";
import React from "react";

const TextHeader = ({ detected_message }) => {
  return (
    <Box sx={{ width: 616, height: 120 }}>
      <Typography
        variant="h1"
        sx={{
          fontFamily: "'Spartan', sans-serif",
          fontWeight: 300,
          WebkitTextStroke: "0.5px black", 
          color: "black",
          fontSize: 40,
          letterSpacing: 0.15,
          lineHeight: "60px",
        }}
      >
        {detected_message}
      </Typography>
    </Box>
  );
};

export default TextHeader;