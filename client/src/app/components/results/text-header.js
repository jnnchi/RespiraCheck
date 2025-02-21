import { Box, Typography } from "@mui/material";
import React from "react";

const Label = () => {
  return (
    <Box sx={{ width: 616, height: 120 }}>
      <Typography
        variant="h1"
        sx={{
          fontFamily: "'Spartan', sans-serif",
          fontWeight: 600,
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

export default Label;
