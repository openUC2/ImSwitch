import React from 'react';

class MJPEGStream extends React.Component {
    render() {
        return (
            <img
                src="{`${hostIP}:${apiPort}/imswitch/api/RecordingController/video_feeder`}"
                alt="MJPEG Stream"
                style={{ width: '100%' }}
            />
        );
    }
}
