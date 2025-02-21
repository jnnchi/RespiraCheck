import { Box, Typography } from "@mui/material";
import React from "react";

const TextHeader = () => {
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
        RespiraCheck has detected a potential for:
      </Typography>
    </Box>
  );
};

export default TextHeader;
