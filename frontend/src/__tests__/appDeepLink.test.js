// The QR codes that carry these URLs get printed and stuck on hardware, so the
// loose matching has to keep working across renames.
import { getDeepLinkApp, readAppParam, resolveApp } from "../utils/appDeepLink";

const BASE = "http://192.168.4.1/imswitch/ui/index.html";

describe("readAppParam", () => {
  it("reads ?app= from the query string", () => {
    expect(readAppParam(`${BASE}?app=holo`)).toBe("holo");
  });

  it("reads ?app= from inside the hash", () => {
    expect(readAppParam(`${BASE}#/?app=holo`)).toBe("holo");
  });

  it("ignores a hash that only carries a route", () => {
    expect(readAppParam(`${BASE}#/mobile`)).toBeNull();
    expect(readAppParam(BASE)).toBeNull();
  });
});

describe("resolveApp", () => {
  it("matches id, pluginId, name and keyword spellings of one app", () => {
    for (const value of [
      "holoController",
      "HoloController",
      "Hologram Processing",
      "hologram-processing",
      "holo",
      "holobox",
      "inline",
    ]) {
      expect(resolveApp(value)?.pluginId).toBe("HoloController");
    }
  });

  it("prefers an exact id over another app's keyword", () => {
    expect(resolveApp("offAxisHoloController")?.pluginId).toBe(
      "OffAxisHoloController",
    );
  });

  it("returns null for junk", () => {
    expect(resolveApp("definitely-not-an-app")).toBeNull();
    expect(resolveApp("")).toBeNull();
    expect(resolveApp(undefined)).toBeNull();
  });
});

describe("getDeepLinkApp", () => {
  it("resolves the app named by the URL", () => {
    expect(getDeepLinkApp(`${BASE}?app=holo`)?.id).toBe("holoController");
  });

  it("is null when no app is requested", () => {
    expect(getDeepLinkApp(BASE)).toBeNull();
  });
});
