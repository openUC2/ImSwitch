// src/i18n/index.js
// Minimal i18n for the ImSwitch frontend.
//
// The lookup key IS the English source string (the gettext model). That buys
// two things over a key registry:
//   * a missing translation renders readable English, never "holo.roi.title"
//   * adding a language means adding one { english: translation } file — there
//     is no key list to keep in sync with the JSX
//
// Adding a language:
//   1. create src/i18n/<code>.js exporting a flat { english: translation } map
//   2. import it into CATALOGS and add it to LANGUAGES below
//
// Usage in a component:
//   const t = useT();
//   <Button>{t("Start")}</Button>
//   <Alert>{t("{percent}% of pixels are saturated.", { percent: 12 })}</Alert>
import { useCallback } from "react";
import { useSelector } from "react-redux";

import de from "./de";
import { getLanguage } from "../state/slices/LanguageSlice";

const CATALOGS = { de };

// Order defines the order of the language switcher in the settings menu.
export const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "de", label: "Deutsch" },
];

export function translate(language, text, vars) {
  let out = CATALOGS[language]?.[text] ?? text;
  if (vars) {
    for (const [name, value] of Object.entries(vars)) {
      out = out.split(`{${name}}`).join(String(value));
    }
  }
  return out;
}

// Returns a stable t(text, vars) bound to the currently selected language.
export function useT() {
  const language = useSelector(getLanguage);
  return useCallback((text, vars) => translate(language, text, vars), [language]);
}
