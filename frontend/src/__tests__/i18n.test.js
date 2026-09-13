// Guards the two ways the string-as-key i18n model can quietly break:
// a placeholder that stops being substituted, and a translated screen that
// drifts back to English because new strings were never added to the catalog.
import fs from "fs";
import path from "path";

import de from "../i18n/de";
import { translate } from "../i18n";

describe("translate", () => {
  it("returns the German string for a known key", () => {
    expect(translate("de", "Documentation")).toBe("Dokumentation");
  });

  it("falls back to the English source for an untranslated string", () => {
    expect(translate("de", "Not in any catalog")).toBe("Not in any catalog");
    expect(translate("en", "Documentation")).toBe("Documentation");
  });

  it("substitutes every occurrence of a placeholder", () => {
    expect(translate("en", "{a} and {a} and {b}", { a: 1, b: 2 })).toBe(
      "1 and 1 and 2",
    );
  });

  it("leaves unknown placeholders alone rather than blanking them", () => {
    expect(translate("en", "{a}/{b}", { a: "x" })).toBe("x/{b}");
  });
});

describe("German catalog coverage", () => {
  // The hologram app is the screen we promised in both languages.
  const source = fs.readFileSync(
    path.join(__dirname, "../components/HoloController.js"),
    "utf8",
  );

  it("covers every literal HoloController passes to t()", () => {
    const literals = [
      ...source.matchAll(/\bt\(\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')/g),
    ].map((m) => (m[1] !== undefined ? m[1] : m[2]).replace(/\\(.)/g, "$1"));

    expect(literals.length).toBeGreaterThan(50); // the regex still finds them
    expect(literals.filter((s) => !(s in de))).toEqual([]);
  });

  it("keeps the placeholders of every translated string", () => {
    const placeholders = (s) => (s.match(/\{[a-zA-Z]+\}/g) || []).sort();
    const broken = Object.entries(de).filter(
      ([en, german]) =>
        placeholders(en).join() !== placeholders(german).join(),
    );
    expect(broken).toEqual([]);
  });
});
