import { Avatar, Box, Typography } from "@mui/material";
import React from "react";

const steps = [
  { number: 1, text: "Upload Audio Or Record", color: "black" },
  { number: 2, text: "Wait for Model Prediction", color: "#3d70ec" },
  { number: 3, text: "View Prediction", color: "#3d70ec" },
];

const Steps = () => {
  return (
    <Box sx={{ position: "relative", width: 565, height: 300, verticalAlign: "center"}}>
      {steps.map((step, index) => (
        <Box
          key={index}
          sx={{
            position: "absolute",
            top: index * 150,
            left: 81,
            display: "flex",
            alignItems: "center",
          }}
        >
          <Avatar
            sx={{
              bgcolor: index === 0 ? "#3d70ec" : "#83a2ee",
              width: 48,
              height: 48,
              marginRight: 2,
            }}
          >
            <Typography
              sx={{
                color: index === 0 ? "white" : "#f1e39b",
                fontFamily: "Roboto",
                fontWeight: "bold",
                fontSize: 32,
              }}
            >
              {step.number}
            </Typography>
          </Avatar>
          <Typography
            sx={{
              fontFamily: "Spartan",
              fontWeight: "bold",
              fontSize: 30,
              color: step.color,
            }}
          >
            {step.text}
          </Typography>
        </Box>
      ))}
    </Box>
  );
};

export default Steps;
