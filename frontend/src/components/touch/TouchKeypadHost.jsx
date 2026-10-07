// src/components/touch/TouchKeypadHost.jsx
//
// On-screen number pad for numeric fields.
//
// A Raspberry Pi touchscreen running chromium in kiosk mode has no OS
// keyboard, so every numeric field in the UI (speeds, exposure, step sizes,
// positions, ...) was effectively read-only there. Rather than rewriting
// hundreds of fields, this host is mounted once in App.jsx and attaches itself
// to any focused
//
//   <input type="number">   or   <input inputmode="numeric|decimal">
//
// while the keypad is enabled (useDeviceProfile().keypad — touch UI on a
// device without its own soft keyboard, or forced in Settings).
//
// The typed value is buffered and written to the field only on OK, the way a
// physical keyboard would deliver it: set the value + `input`/`change` events
// (React's onChange fires), then Enter, then blur. That covers fields that
// commit on change, on Enter and on blur alike, and avoids streaming partial
// values ("2", "20", "200"...) to hardware-backed fields.
//
// Per-field hints (all optional, set via inputProps):
//   data-keypad-presets="1000,5000,10000"   quick-pick chips
//   data-no-keypad                           opt a field (or subtree) out
// min / max / step attributes are honoured.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Box, ButtonBase, Chip, IconButton, Paper, Typography } from "@mui/material";
import { alpha, useTheme } from "@mui/material/styles";
import BackspaceOutlinedIcon from "@mui/icons-material/BackspaceOutlined";
import CloseIcon from "@mui/icons-material/Close";
import RemoveIcon from "@mui/icons-material/Remove";
import AddIcon from "@mui/icons-material/Add";
import useDeviceProfile from "../../hooks/useDeviceProfile";

const KEYPAD_WIDTH = 300;
const MAX_LENGTH = 16;
const ORIG_INPUTMODE = "keypadOrigInputmode"; // dataset key

// ── field inspection ──────────────────────────────────────────────────────

// While the pad is open the field's inputmode is set to "none" (suppresses
// any OS soft keyboard), so read the original value from the dataset first.
const originalInputMode = (el) =>
  el.dataset[ORIG_INPUTMODE] !== undefined
    ? el.dataset[ORIG_INPUTMODE]
    : el.getAttribute("inputmode") || "";

export const isKeypadTarget = (el) =>
  el instanceof HTMLInputElement &&
  !el.readOnly &&
  !el.disabled &&
  (el.type === "number" ||
    ["numeric", "decimal"].includes(originalInputMode(el))) &&
  !el.closest("[data-no-keypad]");

const suppressSoftKeyboard = (el) => {
  if (el.dataset[ORIG_INPUTMODE] !== undefined) return;
  el.dataset[ORIG_INPUTMODE] = el.getAttribute("inputmode") || "";
  el.setAttribute("inputmode", "none");
};

const restoreSoftKeyboard = (el) => {
  const orig = el.dataset[ORIG_INPUTMODE];
  if (orig === undefined) return;
  if (orig) el.setAttribute("inputmode", orig);
  else el.removeAttribute("inputmode");
  delete el.dataset[ORIG_INPUTMODE];
};

const numberAttr = (el, name) => {
  const raw = el.getAttribute(name);
  if (raw === null || raw === "" || raw === "any") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
};

const fieldLabel = (el) => {
  const text =
    el.labels?.[0]?.textContent ||
    el.getAttribute("aria-label") ||
    el.getAttribute("placeholder") ||
    el.name ||
    "";
  return text.replace(/\s*\*\s*$/, "").trim() || "Enter value";
};

// Text of the start/end adornments (e.g. "µm", "ms") next to an MUI input.
// The notched-outline <fieldset> repeats the label, so it is skipped.
const fieldUnit = (el) => {
  const root = el.closest(".MuiInputBase-root");
  if (!root) return "";
  const text = Array.from(root.children)
    .filter((child) => child !== el && child.tagName !== "FIELDSET")
    .map((child) => child.textContent.trim())
    .filter(Boolean)
    .join(" ");
  return text.length <= 10 ? text : "";
};

const decimalsOf = (n) => {
  const s = String(n);
  if (s.includes("e-")) return Number(s.split("e-")[1]);
  return s.includes(".") ? s.split(".")[1].length : 0;
};

// Without a step attribute, nudge by roughly a tenth of the magnitude:
// 20000 -> 1000, 50 -> 1, 0.25 -> 0.01.
const defaultStep = (value) => {
  const v = Math.abs(Number(value));
  if (!Number.isFinite(v) || v === 0) return 1;
  return Math.pow(10, Math.floor(Math.log10(v)) - 1);
};

const formatNumber = (n, step) =>
  String(Number(n.toFixed(Math.min(10, Math.max(decimalsOf(step), 0)))));

// ── writing back into a React-controlled input ────────────────────────────

const nextTick = () => new Promise((resolve) => setTimeout(resolve, 0));

const setNativeValue = (el, value) => {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  ).set;
  // Bypassing React's value tracker is what makes onChange fire.
  setter.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
};

const pressKey = (el, type) =>
  el.dispatchEvent(
    new KeyboardEvent(type, {
      key: "Enter",
      code: "Enter",
      keyCode: 13,
      which: 13,
      bubbles: true,
      cancelable: true,
    }),
  );

// Each step waits a tick so React re-renders in between: a field that commits
// on blur reads its draft from the render *after* the onChange.
async function commitToField(el, value) {
  if (!el.isConnected) return;
  setNativeValue(el, value);
  await nextTick();
  if (!el.isConnected) return;
  pressKey(el, "keydown");
  pressKey(el, "keyup");
  await nextTick();
  if (el.isConnected && document.activeElement === el) el.blur();
}

// ── keys ──────────────────────────────────────────────────────────────────

// Keys must not take focus away from the field: a blur would make
// commit-on-blur fields fire with the old value and close the pad.
const keepFocus = (e) => e.preventDefault();

const REPEAT_DELAY_MS = 450;
const REPEAT_INTERVAL_MS = 90;

function Key({ onPress, repeat = false, tone = "digit", disabled, children, sx, ...rest }) {
  const theme = useTheme();
  const timers = useRef({});

  const stop = () => {
    clearTimeout(timers.current.delay);
    clearInterval(timers.current.interval);
    timers.current = {};
  };
  useEffect(() => stop, []);

  const repeatHandlers = repeat
    ? {
        onPointerDown: (e) => {
          if (disabled || e.button > 0) return;
          onPress();
          stop();
          timers.current.delay = setTimeout(() => {
            timers.current.interval = setInterval(onPress, REPEAT_INTERVAL_MS);
          }, REPEAT_DELAY_MS);
        },
        onPointerUp: stop,
        onPointerLeave: stop,
        onPointerCancel: stop,
      }
    : { onClick: () => !disabled && onPress() };

  const palette = {
    digit: {
      bg: alpha(theme.palette.text.primary, 0.08),
      fg: theme.palette.text.primary,
    },
    action: {
      bg: alpha(theme.palette.text.primary, 0.16),
      fg: theme.palette.text.primary,
    },
    primary: {
      bg: theme.palette.primary.main,
      fg: theme.palette.primary.contrastText,
    },
  }[tone];

  return (
    <ButtonBase
      tabIndex={-1}
      disabled={disabled}
      onMouseDown={keepFocus}
      {...repeatHandlers}
      {...rest}
      sx={{
        minHeight: 48,
        borderRadius: 1.5,
        fontSize: "1.35rem",
        fontWeight: 500,
        fontVariantNumeric: "tabular-nums",
        bgcolor: palette.bg,
        color: palette.fg,
        touchAction: "manipulation",
        "&:active": { filter: "brightness(1.35)" },
        "&.Mui-disabled": { opacity: 0.35 },
        ...sx,
      }}
    >
      {children}
    </ButtonBase>
  );
}

// ── host ──────────────────────────────────────────────────────────────────

export default function TouchKeypadHost() {
  const { keypad } = useDeviceProfile();
  const theme = useTheme();

  const [session, setSession] = useState(null);
  const [buffer, setBuffer] = useState("");
  // The value shown when the pad opens is "selected": the first digit
  // replaces it, like a calculator.
  const [fresh, setFresh] = useState(true);
  const sessionRef = useRef(null);
  sessionRef.current = session;

  // `restore: false` when a commit still has to run against the focused
  // field: putting its inputmode back first could flash a soft keyboard.
  const close = useCallback(({ restore = true } = {}) => {
    const current = sessionRef.current;
    if (current && restore) restoreSoftKeyboard(current.el);
    sessionRef.current = null;
    setSession(null);
  }, []);

  const open = useCallback((el) => {
    if (sessionRef.current?.el === el) return;
    if (sessionRef.current) restoreSoftKeyboard(sessionRef.current.el);
    suppressSoftKeyboard(el);

    const rect = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    // Dock on the side away from the field so it stays visible.
    let dock;
    if (vw >= 640 && vw > vh) {
      dock = rect.left + rect.width / 2 < vw / 2 ? "right" : "left";
    } else {
      dock = rect.top + rect.height / 2 > vh / 2 ? "top" : "bottom";
    }

    const min = numberAttr(el, "min");
    const max = numberAttr(el, "max");
    const stepAttr = numberAttr(el, "step");
    const presets = (el.dataset.keypadPresets || "")
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s !== "" && Number.isFinite(Number(s)));

    const next = {
      el,
      dock,
      label: fieldLabel(el),
      unit: fieldUnit(el),
      min,
      max,
      step: stepAttr && stepAttr > 0 ? stepAttr : defaultStep(el.value),
      presets,
      allowNegative: min === null || min < 0,
      allowDecimal: originalInputMode(el) !== "numeric",
    };
    sessionRef.current = next;
    setSession(next);
    setBuffer(el.value ?? "");
    setFresh(true);
  }, []);

  // Attach to numeric fields while the keypad is enabled.
  useEffect(() => {
    if (!keypad) {
      close();
      return undefined;
    }
    // pointerdown comes before focus: switching the soft keyboard off here
    // stops a phone/tablet keyboard from flashing up behind the pad.
    const onPointerDown = (e) => {
      if (isKeypadTarget(e.target)) suppressSoftKeyboard(e.target);
    };
    const onFocusIn = (e) => {
      if (isKeypadTarget(e.target)) open(e.target);
    };
    // Re-open on a tap into the field that already has focus (after Cancel).
    const onClick = (e) => {
      if (!sessionRef.current && isKeypadTarget(e.target)) open(e.target);
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("click", onClick, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("click", onClick, true);
    };
  }, [keypad, open, close]);

  // Close (without committing) when the field loses focus or disappears.
  useEffect(() => {
    if (!session) return undefined;
    const { el } = session;
    const onFocusOut = () => {
      if (sessionRef.current?.el === el) close();
    };
    el.addEventListener("focusout", onFocusOut);
    const watchdog = setInterval(() => {
      if (!el.isConnected) close();
    }, 500);
    return () => {
      el.removeEventListener("focusout", onFocusOut);
      clearInterval(watchdog);
    };
  }, [session, close]);

  // ── editing ──
  // Partial input while typing is fine: "5." parses as 5; only a lone "-" does
  // not parse yet.
  const trimmed = buffer.trim();
  const parsed = trimmed === "" ? null : Number(trimmed);
  const incomplete = parsed !== null && !Number.isFinite(parsed);
  let error = "";
  if (incomplete) {
    if (trimmed !== "-") error = "Not a number";
  } else if (parsed !== null && session) {
    if (session.min !== null && parsed < session.min) {
      error = `Minimum is ${session.min}`;
    } else if (session.max !== null && parsed > session.max) {
      error = `Maximum is ${session.max}`;
    }
  }

  const typeChars = (chars) => {
    if (chars === "." && !session?.allowDecimal) return;
    setBuffer((prev) => {
      const base = fresh ? "" : prev;
      if (chars === "." && base.includes(".")) return base;
      const next =
        chars === "." && (base === "" || base === "-") ? `${base}0.` : base + chars;
      return next.length > MAX_LENGTH ? base : next;
    });
    setFresh(false);
  };

  const backspace = () => {
    setBuffer((prev) => (fresh ? "" : prev.slice(0, -1)));
    setFresh(false);
  };

  const clear = () => {
    setBuffer("");
    setFresh(false);
  };

  const toggleSign = () => {
    if (!session?.allowNegative) return;
    setBuffer((prev) => (prev.startsWith("-") ? prev.slice(1) : `-${prev}`));
    setFresh(false);
  };

  const nudge = (direction) => {
    const s = sessionRef.current;
    if (!s) return;
    setBuffer((prev) => {
      const current = Number(prev);
      let next = (Number.isFinite(current) ? current : 0) + direction * s.step;
      if (s.min !== null) next = Math.max(s.min, next);
      if (s.max !== null) next = Math.min(s.max, next);
      return formatNumber(next, s.step);
    });
    setFresh(true);
  };

  const pickPreset = (value) => {
    setBuffer(value);
    setFresh(true);
  };

  const confirm = () => {
    const s = sessionRef.current;
    if (!s) return;
    if (buffer.trim() === "") {
      // Nothing typed: leave the field as it was.
      cancel();
      return;
    }
    const value = Number(buffer.trim());
    if (!Number.isFinite(value) || error) return;
    close({ restore: false });
    commitToField(s.el, String(value)).finally(() => restoreSoftKeyboard(s.el));
  };

  const cancel = () => {
    const s = sessionRef.current;
    close({ restore: false });
    if (s?.el) {
      if (document.activeElement === s.el) s.el.blur();
      restoreSoftKeyboard(s.el);
    }
  };

  // A physical keyboard keeps working while the pad is open (USB keyboard on
  // a kiosk, or a desktop with touch mode forced on).
  const handlersRef = useRef({});
  handlersRef.current = { typeChars, backspace, confirm, cancel, toggleSign, nudge };
  useEffect(() => {
    if (!session) return undefined;
    const onKeyDown = (e) => {
      if (!e.isTrusted) return; // our own synthetic Enter from commitToField
      const h = handlersRef.current;
      let handled = true;
      if (/^[0-9]$/.test(e.key)) h.typeChars(e.key);
      else if (e.key === "." || e.key === ",") h.typeChars(".");
      else if (e.key === "-") h.toggleSign();
      else if (e.key === "Backspace") h.backspace();
      else if (e.key === "Enter" || e.key === "Tab") h.confirm();
      else if (e.key === "Escape") h.cancel();
      else if (e.key === "ArrowUp") h.nudge(1);
      else if (e.key === "ArrowDown") h.nudge(-1);
      else handled = false;
      if (handled) {
        e.preventDefault();
        e.stopPropagation();
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [session]);

  if (!session) return null;

  const { dock, label, unit, min, max, step, presets, allowNegative, allowDecimal } = session;
  const side = dock === "left" || dock === "right";
  const stepLabel = formatNumber(step, step);

  const rangeHint =
    min !== null && max !== null
      ? `${min} – ${max}`
      : min !== null
        ? `≥ ${min}`
        : max !== null
          ? `≤ ${max}`
          : "";

  const placement = side
    ? {
        top: 8,
        bottom: 8,
        [dock]: 8,
        width: KEYPAD_WIDTH,
        maxHeight: 520,
        my: "auto",
      }
    : {
        left: 0,
        right: 0,
        [dock]: 0,
        mx: "auto",
        width: "100%",
        maxWidth: 520,
        borderRadius: dock === "bottom" ? "16px 16px 0 0" : "0 0 16px 16px",
      };

  return createPortal(
    <Paper
      elevation={12}
      role="dialog"
      aria-label={`Number pad: ${label}`}
      data-no-keypad
      onMouseDown={keepFocus}
      onContextMenu={(e) => e.preventDefault()}
      sx={{
        position: "fixed",
        zIndex: 1700, // above dialogs (1300), snackbars (1400), tooltips (1500)
        display: "flex",
        flexDirection: "column",
        gap: 1,
        p: 1.25,
        userSelect: "none",
        WebkitUserSelect: "none",
        bgcolor: "background.paper",
        backgroundImage: "none",
        border: 1,
        borderColor: "divider",
        ...placement,
      }}
    >
      {/* header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, minHeight: 32 }}>
        <Typography
          variant="subtitle2"
          sx={{ flex: 1, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
        >
          {label}
          {unit && (
            <Typography component="span" variant="caption" sx={{ ml: 0.75, color: "text.secondary" }}>
              {unit}
            </Typography>
          )}
        </Typography>
        <IconButton size="small" tabIndex={-1} onMouseDown={keepFocus} onClick={cancel} aria-label="Cancel">
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* display */}
      <Box sx={{ display: "flex", alignItems: "stretch", gap: 1 }}>
        <Box
          sx={{
            flex: 1,
            minWidth: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            px: 1.5,
            minHeight: 52,
            borderRadius: 1.5,
            border: 2,
            borderColor: error ? "error.main" : "primary.main",
            bgcolor: alpha(theme.palette.text.primary, 0.04),
            overflow: "hidden",
          }}
        >
          <Typography
            component="span"
            sx={{
              fontSize: "1.6rem",
              fontVariantNumeric: "tabular-nums",
              whiteSpace: "nowrap",
              px: 0.5,
              borderRadius: 0.5,
              bgcolor: fresh && buffer ? alpha(theme.palette.primary.main, 0.3) : "transparent",
              color: buffer ? "text.primary" : "text.disabled",
            }}
          >
            {buffer || "—"}
          </Typography>
        </Box>
        <Key tone="action" onPress={backspace} repeat sx={{ width: 60, flexShrink: 0 }} aria-label="Backspace">
          <BackspaceOutlinedIcon />
        </Key>
      </Box>

      <Typography
        variant="caption"
        sx={{ minHeight: 18, color: error ? "error.main" : "text.secondary" }}
      >
        {error || (rangeHint ? `Range ${rangeHint}` : " ")}
        {!error && fresh && buffer ? (rangeHint ? " · " : "") + "type to replace" : ""}
      </Typography>

      {presets.length > 0 && (
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, flexShrink: 0 }}>
          {presets.map((p) => (
            <Chip
              key={p}
              size="small"
              label={Number(p).toLocaleString()}
              tabIndex={-1}
              onMouseDown={keepFocus}
              onClick={() => pickPreset(p)}
              color={buffer === p ? "primary" : "default"}
              variant={buffer === p ? "filled" : "outlined"}
              sx={{ fontVariantNumeric: "tabular-nums" }}
            />
          ))}
        </Box>
      )}

      {/* keys */}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gridAutoRows: side ? "minmax(48px, 1fr)" : "52px",
          gap: 0.75,
          flex: side ? 1 : "0 0 auto",
          minHeight: 0,
        }}
      >
        {["7", "8", "9"].map((d) => (
          <Key key={d} onPress={() => typeChars(d)}>{d}</Key>
        ))}
        <Key tone="action" repeat onPress={() => nudge(-1)} aria-label={`Minus ${stepLabel}`} sx={{ fontSize: "0.95rem", gap: 0.25 }}>
          <RemoveIcon fontSize="small" />
          {stepLabel}
        </Key>

        {["4", "5", "6"].map((d) => (
          <Key key={d} onPress={() => typeChars(d)}>{d}</Key>
        ))}
        <Key tone="action" repeat onPress={() => nudge(1)} aria-label={`Plus ${stepLabel}`} sx={{ fontSize: "0.95rem", gap: 0.25 }}>
          <AddIcon fontSize="small" />
          {stepLabel}
        </Key>

        {["1", "2", "3"].map((d) => (
          <Key key={d} onPress={() => typeChars(d)}>{d}</Key>
        ))}
        {allowNegative ? (
          <Key tone="action" onPress={toggleSign} aria-label="Toggle sign">±</Key>
        ) : (
          <Key onPress={() => typeChars("00")}>00</Key>
        )}

        <Key tone="action" onPress={clear} sx={{ fontSize: "1rem" }}>C</Key>
        <Key onPress={() => typeChars("0")}>0</Key>
        <Key onPress={() => typeChars(".")} disabled={!allowDecimal}>.</Key>
        <Key tone="primary" onPress={confirm} disabled={Boolean(error) || incomplete} sx={{ fontSize: "1.05rem", fontWeight: 700 }}>
          OK
        </Key>
      </Box>
    </Paper>,
    document.body,
  );
}
