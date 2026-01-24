# Use an official Node.js runtime as a parent image
# This will be specified during the build process to use the correct architecture
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
#FROM arm32v7/node:14 
#FROM node:14


# Clone the app
RUN git clone https://github.com/beniroquai/imswitch-aiortc-react /appMicroscope

# Set the working directory to the cloned repository
WORKDIR /appMicroscope

# Install dependencies for the cloned repository
RUN npm install
RUN npm install @mui/material @emotion/react @emotion/styled

# Replace the homepage string in package.json
RUN sed -i 's|"homepage": "https://youseetoo.github.io/imswitch"|"homepage": "https://youseetoo.github.io/"|' package.json

# Build the application
RUN npm run build

# Install serve globally to serve the build directory
RUN npm install -g serve

# Set the command to serve the app
CMD ["serve", "-s", "build"]

# Expose the port the app runs on
EXPOSE 3000
