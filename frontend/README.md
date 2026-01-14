Note: this repo has been merged into github.com/openUC2/ImSwitch; development and maintenance will continue there.

# React Frontend for ImSwitch

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

This is a first attempt to control an ImSwitch-controlled microscope using the browser using WebRTC implemented in AIORTC.

## Frontend Intensity Scaling

The application now includes frontend intensity scaling capabilities:

- **LiveViewControlWrapper**: Unified image viewer component with frontend intensity scaling
- **Canvas-based processing**: 8-bit JPEG images are processed in the frontend using Canvas API
- **Real-time scaling**: Intensity range sliders (0-255) for immediate visual feedback
- **Scale bar overlay**: Displays actual measurements when pixel size is available
- **Position control overlay**: Integrated stage movement controls

### Usage Example

```javascript
import React from 'react';
import { Provider } from 'react-redux';
import store from '../state/store';
import LiveViewControlWrapper from './axon/LiveViewControlWrapper';

function App() {
  return (
    <Provider store={store}>
      <LiveViewControlWrapper />
    </Provider>
  );
}
```

The component includes:
- Frontend intensity scaling (eliminates backend processing)
- Overlay intensity sliders on the right side
- Scale bar showing measurements in micrometers
- Position controls for stage movement

# Showcase

![](./IMAGES/screencast_2.gif)


# Docker

You can use this inside docker using the following commands:
```bash
sudo docker build -t microscope-app .
sudo docker run -p 5000:5000 8001:8001 microscope-app
```
or pull the herin compiled docker image and run it
```bash
sudo docker pull ghcr.io/openuc2/imswitch-aiortc-react:48332dcb133e7648fe860023dd7657c1323feca8
sudo docker run -p 5000:5000 8001:8001 microscope-app
```

## Available Scripts

In the project directory, you can run:

### `npm install`

Install the package using `npm install`

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

### `python server.py`

start the server that provides the stream and the control endpoints for accessing the stage and illumination values. 

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
