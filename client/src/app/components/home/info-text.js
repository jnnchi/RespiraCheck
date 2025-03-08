import React from "react";
import { Box, Typography } from "@mui/material";
import Link from "next/link";

const InfoText = () => {
  return (
    <Box
      sx={{
        width: 600,
        height: 218,
        position: "relative",
      }}
    >
      <Typography
        sx={{
          fontWeight: 300,
          color: "black",
          textJustify: "left",
          fontSize: 19,
        }}
      >
        RespiraCheck is a COVID-19 testing tool that classifies your cough as
        potentially positive or negative for COVID. By bridging the gap between
        clinical research and practical deployment, we aim to provide an
        accessible, non-invasive, and scalable tool for COVID-19 screening.
      </Typography>
      <Typography
        sx={{
          paddingTop: "10px",
          fontWeight: 100,
          color: "black",
          textJustify: "left",
          fontSize: 19,
          textDecoration: "underline",
          textDecorationThickness: 0.8,
        }}
      >
        <Link href="/pages/about">
          <span style={{ textDecoration: "none" }}>Learn More &gt;</span>
        </Link>
      </Typography>
    </Box>
  );
};

export default InfoText;
