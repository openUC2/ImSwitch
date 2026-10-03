const { ModuleFederationPlugin } = require("webpack").container;

module.exports = {
  // uuid (and a few other deps) ship ESM only; CRA's jest transform skips
  // node_modules by default, so importing anything that reaches them fails to
  // parse. Allow-list them so the modules under test can actually be loaded.
  jest: {
    configure: (config) => {
      config.transformIgnorePatterns = [
        "[/\\\\]node_modules[/\\\\](?!(uuid|nanoid|axios)[/\\\\])",
        "^.+\\.module\\.(css|sass|scss)$",
      ];
      return config;
    },
  },
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
          shared: {
            react: { singleton: true, eager: true, requiredVersion: false },
            "react-dom": {
              singleton: true,
              eager: true,
              requiredVersion: false,
            },
            "react/jsx-runtime": {
              singleton: true,
              eager: true,
              requiredVersion: false,
            },
          },
        })
      );

      return config;
    },
  },
};
