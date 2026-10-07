import { resolveDeviceProfile } from "../hooks/useDeviceProfile";

const env = (overrides = {}) => ({
  isTouch: false,
  hasTouch: false,
  isMobileOS: false,
  widthClass: "xl",
  isShort: false,
  isPortrait: false,
  ...overrides,
});
const auto = { touchMode: "auto", onScreenKeypad: "auto" };

describe("resolveDeviceProfile", () => {
  test("desktop with a mouse keeps the desktop UI", () => {
    const p = resolveDeviceProfile(env(), auto);
    expect(p).toMatchObject({ touchUI: false, compactLayout: false, keypad: false });
  });

  test("Raspberry Pi 800x480 touchscreen: touch UI, compact layout, number pad", () => {
    const p = resolveDeviceProfile(
      env({ isTouch: true, hasTouch: true, widthClass: "sm", isShort: true }),
      auto,
    );
    expect(p).toMatchObject({ touchUI: true, compactLayout: true, keypad: true });
  });

  test("phones bring their own keyboard, so the pad stays off on auto", () => {
    const p = resolveDeviceProfile(
      env({ isTouch: true, hasTouch: true, isMobileOS: true, widthClass: "xs", isPortrait: true }),
      auto,
    );
    expect(p).toMatchObject({ touchUI: true, compactLayout: true, phoneLayout: true, keypad: false });
  });

  test("a mid-size touch tablet gets the compact layout, a mid-size desktop does not", () => {
    expect(resolveDeviceProfile(env({ isTouch: true, widthClass: "md" }), auto).compactLayout).toBe(true);
    expect(resolveDeviceProfile(env({ widthClass: "md" }), auto).compactLayout).toBe(false);
  });

  test("forcing touch mode on a mouse kiosk enables touch UI and the pad", () => {
    const p = resolveDeviceProfile(env(), { touchMode: "on", onScreenKeypad: "auto" });
    expect(p).toMatchObject({ touchUI: true, keypad: true });
  });

  test("touch mode off restores the desktop layout everywhere", () => {
    const p = resolveDeviceProfile(
      env({ isTouch: true, widthClass: "xs" }),
      { touchMode: "off", onScreenKeypad: "auto" },
    );
    expect(p).toMatchObject({ touchUI: false, compactLayout: false, keypad: false });
  });

  test("the pad can be forced on or off independently", () => {
    expect(resolveDeviceProfile(env(), { touchMode: "auto", onScreenKeypad: "on" }).keypad).toBe(true);
    expect(
      resolveDeviceProfile(env({ isTouch: true }), { touchMode: "auto", onScreenKeypad: "off" }).keypad,
    ).toBe(false);
  });
});
