"use client";

import { Box, Typography } from "@mui/material";
import React from "react";


const Button = () => {
  return (
    <Box
      sx={{
        position: "relative",
        width: "100%",
        height: "154px",
        backgroundColor: "white",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 63px",
      }}
    >
      <Box
        component="img"
        sx={{
          width: "351px",
          height: "73px",
          objectFit: "cover",
        }}
        alt="Screenshot"
        src={''}
      />

      <Typography
        sx={{
          fontSize: "35px",
          color: "black",
          lineHeight: "52.5px",
        }}
      >
        Home
      </Typography>

      <Typography
        sx={{
          fontSize: "35px",
          color: "black",
          lineHeight: "52.5px",
        }}
      >
        Info
      </Typography>

      <Typography
        sx={{
          fontSize: "35px",
          color: "#3d70ec",
          lineHeight: "52.5px",
        }}
      >
        RespiraChecker
      </Typography>
    </Box>
  );
};

export default Button;
