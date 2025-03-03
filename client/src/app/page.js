"use client";

import { React, useEffect, useState } from "react";
import Navbar from "./components/navbar";
import { Stack, Box } from "@mui/material";
import Title from "@/app/components/home/title";
import StartButton from "@/app/components/home/button";
import InfoText from "@/app/components/home/info-text";
import Image from "next/image";
import Link from "next/link";

const Home = () => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <div>
      {isClient ? (
        <>
          <Navbar />
          <Stack
            direction="row"
            spacing={0}
            sx={{ paddingLeft: "120px", paddingTop: "50px" }}
          >
            <Stack direction="column" spacing={5}>
              <Title />
              <StartButton />
              <InfoText />
            </Stack>
            <Box
              sx={{
                paddingRight: "120px",
                paddingTop: "50px",
                alignItems: "flex-start",
              }}
            >
              <Image
                src="/undraw_medical_icon.svg"
                alt="medical image"
                width={500}
                height={367}
              />
            </Box>
          </Stack>
        </>
      ) : (
        <p>Loading...</p> // Prevents mismatched server/client rendering
      )}
    </div>
  );
};

export default Home;
