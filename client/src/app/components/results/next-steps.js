import { Box, Typography } from "@mui/material";
import React from "react";

const NextSteps = () => {
    const nextSteps = [
        {
          recommendations: [
            "Monitor your health for any changes.",
            "Follow public health guidelines on testing and isolation.",
            "Consult a healthcare professional if symptoms develop."
          ]
        },
        {
          recommendations: [
            "Take a COVID-19 test to confirm your status.",
            "Self-isolate and monitor your symptoms.",
            "Follow local public health guidelines on testing and quarantine.",
            "Contact a healthcare provider if you experience severe symptoms."
          ]
        },
        {
          disclaimer: "⚠ This tool is for informational purposes only and does not replace medical testing or professional diagnosis. Always follow official health recommendations."
        }
    ];

    const prediction = localStorage.getItem("prediction");
    return (
        <Box sx={{ paddingTop: 0, position: "relative", width: "100%", maxWidth: 1123, height: 0}}>
            <Typography
                variant="h2"
                component="div"
                sx={{
                    position: "relative",
                    top: 0,
                    left: 0,
                    fontFamily: "Spartan-SemiBold, Helvetica",
                    fontWeight: 500,
                    color: "black",
                    fontSize: "30px",
                }}
            >
                Next Steps:
            </Typography>
            <List sx={{ paddingLeft: 2 }}>
                {nextSteps[prediction].map((recommendation, index) => (
                <ListItem key={index} sx={{ paddingLeft: 0 }}>
                    <ListItemText
                        primary={recommendation}
                        sx={{
                        fontFamily: "Spartan-Regular, Helvetica",
                        color: "black",
                        lineHeight: "30px"
                        }}
                    />
                </ListItem>
                ))}
            </List>

        </Box>
    );
};

export default NextSteps;