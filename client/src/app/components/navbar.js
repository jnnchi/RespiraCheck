"use client";

import { Link } from 'react-router-dom';
import { Box, Typography } from "@mui/material";
import React from "react";
import Image from "next/image";

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
        
      }}
    >

      <Image src="/RespiraCheckLogo.png" width={200} height={80} alt="hi" sx={{justifyContent: 'left'}}></Image>
        <Box sx={{ width: "100%", gap: 15, display: 'flex', justifyContent: "right"}}>
          <Typography
            sx={{
              fontSize: "25px",
              color: "black",
              lineHeight: "52.5px",
            }}
          >
            <Link to="/home" style={{ textDecoration: "none", color: "black" }}>
              Home
            </Link>
            
          </Typography>

          <Typography
            sx={{
              fontSize: "25px",
              color: "black",
              lineHeight: "52.5px",
            }}
          >
            <Link to="/info" style={{ textDecoration: "none", color: "black" }}>
              Info
            </Link>
            
          </Typography>

          <Typography
            sx={{
              fontSize: "25px",
              color: "#3d70ec",
              lineHeight: "52.5px",
            }}
          >
            <Link to="/action" style={{ textDecoration: "none", color: "black" }}>
              RespiraChecker
            </Link>
            
          </Typography>

        </Box>
      
    </Box>
  );
};

export default Navbar;
