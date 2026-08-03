const { ModuleFederationPlugin } = require("webpack").container;
const { makeShared } = require("./shared-deps");

module.exports = {
  webpack: {
    configure: (config) => {
      config.output.publicPath = "/imswitch/ui/";

      // Fix ES module resolution issues with luma.gl
      config.module.rules.push({
        test: /\.m?js$/,
        resolve: {
          fullySpecified: false, // disable the behaviour
        },
      });

      config.plugins.push(
        new ModuleFederationPlugin({
          name: "host_app",

          // The host is a bidirectional federation container: it consumes
          // plugin remotes AND publishes its own modules back to them, so a
          // plugin can reach the real store and the real contexts instead of
          // being handed them as props.
          //
          // Served at /imswitch/ui/remoteEntry.js (config.output.publicPath
          // above). Plugins declare:
          //   remotes: { host_app: "host_app@/imswitch/ui/remoteEntry.js" }
          filename: "remoteEntry.js",
          exposes: {
            "./store": "./src/state/store.js",
            "./contexts": "./src/context/index.js",
            "./sharedDeps": "./shared-deps.js",
          },

          // eager: true because the host is the *provider* of every shared
          // module — they must be present in its bundle at startup. Remotes
          // must use eager: false; see frontend/shared-deps.js.
          shared: makeShared({ eager: true }),
        })
      );

      return config;
    },
  },
};
