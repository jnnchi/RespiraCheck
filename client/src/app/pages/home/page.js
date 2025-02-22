import React from 'react';

import Navbar from '../../components/navbar';
import Link from 'next/link';
import { Stack, Box } from '@mui/material';
import Title from '@/app/components/home/title';
import StartButton from '@/app/components/home/button';
import InfoText from '@/app/components/home/info-text';
import Image from 'next/image';


const Home = () => {
  return (
    <div>
      <Navbar></Navbar>
      <Stack direction="row" spacing={0} sx={{paddingLeft: "120px", paddingTop: "50px"}}>
        <Stack direction="column" spacing={5}>
          <Title/>
          <StartButton/>
          <InfoText/>
        </Stack>
        <Box sx={{paddingRight: "120px", paddingTop: "50px", alignItems: "flex-start"}}>
          <Image src="/undraw_medical_icon.svg" alt="medical image" width={500} height={367} s/> 

        </Box>
      </Stack>

      <Link href="/pages/results"> 
        <span style={{ textDecoration: "underline"}}>Results</span>
      </Link>
    </div>
  );
};

export default Home;
