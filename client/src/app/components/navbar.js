"use client";

import Link from 'next/link';

import { Box, Typography } from "@mui/material";
import React from "react";
import Image from "next/image";

import { usePathname, useRouter } from "next/navigation";

const Navbar = () => {

  const pathname = usePathname();

  return (
    <Box
      sx={{
        position: "relative",
        width: "100%",
        height: "80px",
        backgroundColor: "white",
        display: "flex",
        alignItems: "center",
        padding: "0 35px",
        
      }}
    >

      <Link href="/">
        <Image src="/RespiraCheckLogo.png" width={200} height={80} alt="hi" sx={{display: 'flex', 
        justifyContent: 'flex-start', marginLeft: 0}}></Image>
      </Link>
      <Box sx={{ width: "100%", gap: 15, display: 'flex', justifyContent: "right"}}>
        <Typography
          sx={{
            fontSize: "20px",
            lineHeight: "52.5px",
            fontWeight: 300,
          }}
        >
          <Link href="/"> 
            <span style={{ textDecoration: "none", color: pathname === "/" ? "#3D70EC" : "black" }} 
              onMouseEnter={(e) => e.target.style.color = "#3D70EC"} // Hover color change
              onMouseLeave={(e) => e.target.style.color = pathname === "/" ? "#3D70EC" : "black"}
            >Home</span>
          </Link>
          
        </Typography>

        <Typography
          sx={{
            fontSize: "20px",
            lineHeight: "52.5px",
            fontWeight: 300,
          }}
        >
          <Link href="/pages/about"> 
            <span style={{ textDecoration: "none", color: pathname === "/pages/about" ? "#3D70EC" : "black" }}
              onMouseEnter={(e) => e.target.style.color = "#3D70EC"} 
              onMouseLeave={(e) => e.target.style.color = pathname === "/pages/about" ? "#3D70EC" : "black"}
            >About</span>
          </Link>
          
        </Typography>

        <Typography
          sx={{
            fontSize: "20px",
            lineHeight: "52.5px",
            fontWeight: 300,
          }}
        > 
          <Link href="/pages/use-the-tool"> 
            <span style={{ textDecoration: "none", color: pathname === "/pages/use-the-tool" ? "#3D70EC" : "black" }}
            onMouseEnter={(e) => e.target.style.color = "#3D70EC"} // Hover color change
            onMouseLeave={(e) => e.target.style.color = pathname === "/pages/use-the-tool" ? "#3D70EC" : "black"}
            >Use The Tool</span>
          </Link>
          
        </Typography>

      </Box>
      
    </Box>
  );
};

export default Navbar;
