function loadRemote({ remote, scope, exposed }) {
    console.log("Initializing loadRemote:", { remote, scope, exposed });
    const url = `${hostIP}:${apiPort}${remote}`;
    console.log("Remote URL:", url);
  
    return new Promise((resolve, reject) => {
      if (!document.querySelector(`script[data-mf="${scope}"]`)) {
        const el = document.createElement("script");
        el.src = url;
        el.dataset.mf = scope;
        el.onload = init;
        el.onerror = (error) => {
          console.error("Failed to load remote script:", error);
          reject(error);
        };
        document.head.appendChild(el);
      } else {
        init();
      }
  
      async function init() {
        try {
          // Ensure Webpack sharing is initialized
          if (typeof __webpack_init_sharing__ === "function") {
            await __webpack_init_sharing__("default");
          } else {
            throw new Error("__webpack_init_sharing__ is not defined");
          }
  
          const container = window[scope];
          if (!container) {
            throw new Error(`Container for scope "${scope}" not found`);
          }
  
          await container.init(__webpack_share_scopes__.default);
  
          const factory = await container.get(
            exposed.startsWith("./") ? exposed : `./${exposed}`
          );
          const module = factory();
          if (!module.default) {
            throw new Error("Module does not export a default React component.");
          }
          resolve(module.default);
        } catch (error) {
          console.error("Error during module initialization:", error);
          reject(error);
        }
      }
    });
  }