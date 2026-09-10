// WP-15: the size estimate was always shown but never compared to the drive it
// would be written to, so a run could only fail once the disk filled up.
import { checkFitsOnDrive } from "../axon/experiment-designer/ExperimentSummary";

const GB = 1024 * 1024 * 1024;
const drive = (freeBytes, extra = {}) => ({
  label: "SSD",
  usage: { free: freeBytes },
  ...extra,
});

describe("checkFitsOnDrive", () => {
  it("says a run fits when there is room", () => {
    const result = checkFitsOnDrive(10 * 1024, drive(100 * GB));
    expect(result.fits).toBe(true);
    expect(result.freeText).toBe("100.0 GB");
    expect(result.name).toBe("SSD");
  });

  it("says a run does not fit when it is bigger than the free space", () => {
    expect(checkFitsOnDrive(120 * 1024, drive(100 * GB)).fits).toBe(false);
  });

  it("treats an exactly-full drive as fitting", () => {
    expect(checkFitsOnDrive(100 * 1024, drive(100 * GB)).fits).toBe(true);
  });

  it("returns null when free space is unknown, rather than guessing", () => {
    expect(checkFitsOnDrive(1000, undefined)).toBeNull();
    expect(checkFitsOnDrive(1000, { label: "SSD" })).toBeNull();
    expect(checkFitsOnDrive(1000, { usage: {} })).toBeNull();
    expect(checkFitsOnDrive(1000, { usage: { free: null } })).toBeNull();
  });

  it("falls back to the path, then a generic name, when unlabelled", () => {
    expect(checkFitsOnDrive(1, { usage: { free: GB }, path: "/mnt/ssd" }).name)
      .toBe("/mnt/ssd");
    expect(checkFitsOnDrive(1, { usage: { free: GB } }).name)
      .toBe("the active drive");
  });
});
