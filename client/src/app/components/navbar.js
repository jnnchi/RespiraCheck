"use client";

import { Box, Typography } from "@mui/material";
import React from "react";


const Navbar = () => {
  return (
    <Box
      sx={{
        position: "relative",
        width: "100%",
        height: "80px",
        backgroundColor: "white",
        display: "flex",
        alignItems: "center",
        padding: "0 63px",
        gap: 15,
        justifyContent: "right",
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
          fontSize: "25px",
          color: "black",
          lineHeight: "52.5px",
        }}
      >
        Home
      </Typography>

      <Typography
        sx={{
          fontSize: "25px",
          color: "black",
          lineHeight: "52.5px",
        }}
      >
        Info
      </Typography>

      <Typography
        sx={{
          fontSize: "25px",
          color: "#3d70ec",
          lineHeight: "52.5px",
        }}
      >
        RespiraChecker
      </Typography>
    </Box>
  );
};

export default Navbar;
