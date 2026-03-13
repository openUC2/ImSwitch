import { normalizeStorageSnapshot } from "../state/slices/StorageSlice";

describe("normalizeStorageSnapshot", () => {
  test("normalizes usage values from gigabytes to bytes", () => {
    const snapshot = {
      active_path: "/media/usb1/ImSwitchData",
      active_data_path: "/media/usb1/ImSwitchData",
      storage_devices: [
        {
          path: "/media/usb1",
          is_active: true,
          exists: true,
          usage: {
            free_gb: 50,
            total_gb: 100,
          },
        },
      ],
    };

    const normalized = normalizeStorageSnapshot(snapshot);
    const device = normalized.storage_devices[0];

    expect(device.usage.free).toBe(50 * 1024 ** 3);
    expect(device.usage.total).toBe(100 * 1024 ** 3);
    expect(device.usage.used).toBe(50 * 1024 ** 3);
    expect(device.usage.percent_used).toBe(50);
  });

  test("falls back to an available default/internal device when active path device is unavailable", () => {
    const snapshot = {
      active_path: "/media/removed/ImSwitchData",
      active_data_path: "/media/removed/ImSwitchData",
      storage_devices: [
        {
          path: "/home/user/ImSwitchConfig/data",
          is_internal: true,
          is_default: true,
          exists: true,
        },
        {
          path: "/media/removed",
          is_internal: false,
          exists: false,
        },
      ],
    };

    const normalized = normalizeStorageSnapshot(snapshot);

    expect(normalized.active_path).toBe("/home/user/ImSwitchConfig/data");
    expect(normalized.active_data_path).toBe("/home/user/ImSwitchConfig/data");
    expect(normalized.active_device?.path).toBe(
      "/home/user/ImSwitchConfig/data",
    );
    expect(normalized.active_device_path).toBe(
      "/home/user/ImSwitchConfig/data",
    );
  });

  test("preserves active path when it is valid for an available device", () => {
    const snapshot = {
      active_path: "/media/usb1/ImSwitchData",
      active_data_path: "/media/usb1/ImSwitchData",
      storage_devices: [
        {
          path: "/home/user/ImSwitchConfig/data",
          is_internal: true,
          is_default: true,
          exists: true,
        },
        {
          path: "/media/usb1",
          is_internal: false,
          exists: true,
        },
      ],
    };

    const normalized = normalizeStorageSnapshot(snapshot);

    expect(normalized.active_path).toBe("/media/usb1/ImSwitchData");
    expect(normalized.active_data_path).toBe("/media/usb1/ImSwitchData");
    expect(normalized.active_device?.path).toBe("/media/usb1");
    expect(normalized.active_device_path).toBe("/media/usb1");
  });

  test("keeps only available non-internal devices in external_devices", () => {
    const snapshot = {
      active_path: "/home/user/ImSwitchConfig/data",
      active_data_path: "/home/user/ImSwitchConfig/data",
      storage_devices: [
        {
          path: "/home/user/ImSwitchConfig/data",
          is_internal: true,
          is_default: true,
          exists: true,
        },
        {
          path: "/media/usb1",
          is_internal: false,
          exists: true,
        },
        {
          path: "/media/usb2",
          is_internal: false,
          exists: false,
        },
      ],
    };

    const normalized = normalizeStorageSnapshot(snapshot);

    expect(normalized.external_devices).toHaveLength(1);
    expect(normalized.external_devices[0].path).toBe("/media/usb1");
  });

  test("uses boundary-aware path matching for active device selection", () => {
    const snapshot = {
      active_path: "/media/USB2/ImSwitchData",
      active_data_path: "/media/USB2/ImSwitchData",
      storage_devices: [
        {
          path: "/media/USB",
          is_internal: false,
          exists: true,
        },
        {
          path: "/media/USB2",
          is_internal: false,
          exists: true,
        },
      ],
    };

    const normalized = normalizeStorageSnapshot(snapshot);

    const usb = normalized.storage_devices.find(
      (device) => device.path === "/media/USB",
    );
    const usb2 = normalized.storage_devices.find(
      (device) => device.path === "/media/USB2",
    );

    expect(usb?.is_active).toBe(false);
    expect(usb2?.is_active).toBe(true);
    expect(normalized.active_device?.path).toBe("/media/USB2");
  });
});
