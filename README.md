# RespiraCheck

## Setting up Docker

### **Install Docker Desktop**
1. Access the following [link](https://docs.docker.com/desktop/?_gl=1*1hmw18s*_gcl_au*MTEwMDI1OTAwOC4xNzM4NTM1OTQ0*_ga*MTI0NjUzOTY3NC4xNzM4NTM1OTQ0*_ga_XJWPQMJYHQ*MTczODUzNTk0My4xLjEuMTczODUzNTk0NC41OS4wLjA.) to install.

2. Verify installation by running:
    ```sh
    docker --version
    ```

3. Open Docker Desktop and wait for it to start. This will give you a visual of whether your containers are running.

### **Build and Start the Docker Containers**
1. Run the following command to build the images and start the containers in detached mode:
    ```sh
    docker-compose up --build -d
    ```
    * The `--build` flag ensures the images are rebuilt.
    * The `-d` flag runs the containers in the background (detached mode).

### **Check Running Containers**
3. To see which containers are running, use:
    ```sh
    docker ps
    ```

4. You may need to check logs for the ML container to get the token for the Jupyter URL. Type the following command, and look for the URLs containing the token.
    ```sh
    docker logs <container_id>
    ```

5. To open the FastAPI backend, visit: http://localhost:8000

6. To open the ML folder in Jupyter in your browser, visit: http://localhost:8888/login?next=%2Ftree%3F and paste the token you found from Docker logs in **Step 4** in as a password. Alternatively, simply follow the URL you found in **Step 4**.

7. When you're done with the Docker containers, run:
    ```sh
    docker-compose down
    ```