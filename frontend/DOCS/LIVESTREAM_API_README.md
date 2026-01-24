# LiveStream API Update - README

## 📋 Overview

This update migrates the ImSwitch-ReactAPP frontend from the old `SettingsController` and `ViewController` endpoints to the new unified `LiveViewController` API for managing live streams.

## 🎯 Scope

**Focus**: Binary and JPEG streaming only (as requested)  
**Date**: October 20, 2025  
**Status**: ✅ Implementation complete, ready for testing

## 📦 What's New

### New API Functions (5)
All located in `src/backendapi/`:
- `apiLiveViewControllerGetActiveStreams.js`
- `apiLiveViewControllerGetStreamParameters.js`
- `apiLiveViewControllerSetStreamParameters.js`
- `apiLiveViewControllerStartLiveView.js`
- `apiLiveViewControllerStopLiveView.js`

### Updated Components (4)
- `StreamControlOverlay.js` - Main settings overlay
- `StreamSettings.js` - Alternative settings panel
- `LiveView.js` - Stream control logic
- `apiViewControllerGetLiveViewActive.js` - Documentation update

## 🚀 Quick Start

### For Users
Nothing changes! The UI and workflow remain identical. All improvements are under the hood.

### For Developers

#### Install dependencies
```bash
npm install
```

#### Start development server
```bash
npm start
```

#### Access the app
Open [http://localhost:3000](http://localhost:3000)

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **LIVESTREAM_API_SUMMARY.md** | 📄 Quick summary of changes |
| **LIVESTREAM_API_UPDATE.md** | 📚 Detailed implementation guide |
| **LIVESTREAM_API_QUICKREF.md** | ⚡ Quick reference for developers |
| **LIVESTREAM_API_ARCHITECTURE.md** | 🏗️ System architecture diagrams |
| **LIVESTREAM_API_TESTING.md** | ✅ Testing checklist |

### Start here
- **New to this update?** → Read `LIVESTREAM_API_SUMMARY.md`
- **Need to use the API?** → Read `LIVESTREAM_API_QUICKREF.md`
- **Troubleshooting?** → Check `LIVESTREAM_API_UPDATE.md`
- **Testing the update?** → Use `LIVESTREAM_API_TESTING.md`

## 🔧 Technical Details

### Old Endpoints (Deprecated)
```
GET /SettingsController/getStreamParams
POST /SettingsController/setStreamParams
GET /ViewController/setLiveViewActive
```

### New Endpoints (Current)
```
GET /LiveViewController/getStreamParameters
GET /LiveViewController/setStreamParameters
POST /LiveViewController/startLiveView
GET /LiveViewController/stopLiveView
GET /LiveViewController/getLiveViewActive
GET /LiveViewController/getActiveStreams
```

## 💡 Key Features

### Protocol-Specific Parameters
Each streaming protocol (binary, JPEG, MJPEG, WebRTC) now has its own parameter set:

**Binary**
- Compression algorithm (lz4, zstd, none)
- Compression level (0-9)
- Subsampling factor (1-8)
- Throttle time (16-1000ms)

**JPEG**
- Quality (1-100)
- Subsampling factor (1-8)
- Throttle time (16-1000ms)

### Better Separation of Concerns
- **Global settings**: Configure defaults for each protocol
- **Per-stream settings**: Override defaults when starting a stream
- **Multiple detectors**: Support for streaming from different cameras

## 🧪 Testing

### Run the test checklist
See `LIVESTREAM_API_TESTING.md` for a comprehensive testing guide.

### Quick smoke test
1. Start the app
2. Click "Play" to start stream
3. Open StreamControlOverlay (settings icon)
4. Change compression algorithm
5. Click "Submit"
6. Stop and restart stream
7. Verify new settings are applied

## 🐛 Known Issues

None at this time. Please report issues to the development team.

## ⚠️ Breaking Changes

### For Custom Code
If you have custom components using the old endpoints, they need to be updated:

**Before:**
```javascript
await fetch('/SettingsController/setStreamParams', {
  method: 'POST',
  body: JSON.stringify(params)
});
```

**After:**
```javascript
import apiLiveViewControllerSetStreamParameters from './backendapi/apiLiveViewControllerSetStreamParameters';

await apiLiveViewControllerSetStreamParameters('binary', {
  compression_algorithm: 'lz4',
  compression_level: 0,
  subsampling_factor: 4,
  throttle_ms: 50
});
```

See `LIVESTREAM_API_QUICKREF.md` for more migration examples.

## 🔮 Future Enhancements

### Planned
- MJPEG streaming UI integration
- WebRTC streaming support
- Multi-detector UI support
- Real-time active stream monitoring
- Stream quality metrics display

### Under Consideration
- Stream reconnection handling
- Bandwidth adaptation
- Parameter presets/profiles
- Stream recording to different formats
- Multi-protocol simultaneous streaming

## 📞 Support

### Questions?
1. Check the documentation files (see table above)
2. Review the code comments in API functions
3. Check the browser console for errors
4. Review the Network tab in DevTools

### Report Issues
Include:
- Description of the problem
- Steps to reproduce
- Browser and version
- Console errors (if any)
- Network requests (if relevant)

## 🤝 Contributing

### Code Style
- Follow existing patterns in the codebase
- Add JSDoc comments to all functions
- Include usage examples in comments
- Keep functions focused and single-purpose

### Testing
- Test locally before submitting
- Run through the testing checklist
- Verify no regressions in existing features
- Document any new test cases

## 📜 License

Same as the main ImSwitch-ReactAPP project.

## 🙏 Acknowledgments

This update aligns the frontend with the new LiveViewController backend architecture, providing better flexibility and separation of concerns for future enhancements.

---

**Last Updated**: October 20, 2025  
**Version**: 1.0.0  
**Status**: ✅ Ready for testing
