import React from 'react';

import Navbar from '../../components/navbar';
import Link from 'next/link';


const Home = () => {
  return (
    <div>
      <Navbar></Navbar>
      <h1>Welcome to the Home Page</h1>
      <Link href="/pages/results"> 
        <span style={{ textDecoration: "underline"}}>Results</span>
      </Link>
    </div>
  );
};

export default Home;
