import { Box, Typography } from "@mui/material";
import React from "react";

const NextSteps = ({prediction}) => {
    const nextSteps = prediction === "1" ? "If you have been diagnosed with COVID-19, it is essential to take immediate and responsible action. Self-isolate at home and follow the advice of your healthcare provider closely. Monitor your symptoms frequently and seek medical attention if you experience difficulty breathing, chest pain, or a significant worsening of your condition. Stay well-hydrated, get plenty of rest, and consider using over-the-counter medications to manage mild symptoms after consulting with your doctor. Inform close contacts about your diagnosis so they can monitor their own health, and adhere strictly to quarantine guidelines to prevent further spread. Keep up with local health advisories and remain in regular contact with your healthcare team for ongoing guidance." : 
      "Keep monitoring your symptoms, remember that this is not an official diagnosis and RespiraCheck may make mistakes. Even if you do not have COVID, try to avoid crowded areas and wear a mask to protect yourself and others. Drink lots of water. Continue to practice good hygiene, maintain social distancing, and avoid unnecessary exposure in public spaces. Monitor your temperature and other symptoms closely, and consider getting a COVID-19 test if any new or worsening symptoms appear. Staying informed with the latest public health guidelines and maintaining a healthy lifestyle—including regular exercise and a balanced diet—can also help protect your overall well-being.";
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

            <Typography
                variant="body1"
                component="p"
                sx={{
                    position: "relative",
                    top: 15, 
                    left: 0, 
                    fontFamily: "Spartan-Regular, Helvetica",
                    color: "black",
                    lineHeight: "30px",
                    fontSize: "20px",
                }}
            >
                {nextSteps}
            </Typography>
        </Box>
    );
};

export default NextSteps;