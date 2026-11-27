// src/components/holo/holoGuides.js
// Content of the HoloBox build guides (English source strings; the German
// text lives in i18n/de.js). Photos are in public/assets/holobox, downscaled
// from the openUC2 docs (docs/usage/disc/holobox).

// Cube grid (top view). cells: [col, row, label]; beam: polyline in cell units
// (0.5 = centre of the first cube).
export const LAYOUTS = {
  inline: {
    cols: 4,
    rows: 1,
    cells: [[0, 0, "LED / laser"], [1, 0, "empty"], [2, 0, "empty"], [3, 0, "sample + camera"]],
    beams: [[[0.5, 0.5], [3.5, 0.5]]],
  },
  michelson: {
    cols: 3,
    rows: 3,
    cells: [[0, 1, "fibre (laser)"], [1, 1, "beam splitter"], [2, 1, "mirror (kinematic)"], [1, 0, "mirror"], [1, 2, "screen / camera"]],
    beams: [[[0.5, 1.5], [2.5, 1.5]], [[1.5, 1.5], [1.5, 0.5]], [[1.5, 1.5], [1.5, 2.5]]],
  },
  machzehnder: {
    cols: 4,
    rows: 2,
    cells: [[0, 1, "fibre (laser)"], [1, 1, "beam splitter 1"], [2, 1, "mirror (kinematic)"], [1, 0, "mirror"], [2, 0, "beam splitter 2"], [3, 0, "camera"]],
    beams: [[[0.5, 1.5], [2.5, 1.5], [2.5, 0.5]], [[1.5, 1.5], [1.5, 0.5], [3.5, 0.5]]],
  },
};

export const GUIDES = {
  inline: {
    title: "In-line holography: a lensless microscope",
    intro:
      "A point of light, a sparse sample and a camera - no lens. The light that passes the sample unchanged (primary wave) interferes with the light the sample scatters. The camera records the rings; the computer refocuses them.",
    parts: [
      "LED cube (Holo target) or fibre-coupled red laser in its fibre holder",
      "Raspberry Pi camera module with the sample holder",
      "a sparse sample: dust, pollen or fine sand on a coverslip",
      "base plate and puzzle pieces, 2 empty cubes as spacers",
    ],
    steps: [
      ["inline_1_plate.jpg", "Start with an empty base plate."],
      ["inline_2_parts.jpg", "Put the light source cube at one end. The light has to travel straight along the row."],
      ["inline_3_puzzle.jpg", "Add the camera cube at the other end. Fix every cube with puzzle pieces on top and bottom so the row cannot bend."],
      ["inline_4_laser.jpg", "Put the sample into the camera cube, as close to the sensor as you can - almost touching. A short sample-to-sensor distance gives clear rings even with the partially coherent LED."],
      ["inline_5_done.jpg", "Switch the light on, cover the setup against room light and set the exposure to about 1 ms. You should see shadows with rings around them."],
    ],
    tips: [
      "Dust is the best first sample: small, sparse, high contrast.",
      "Washed-out image: stray light. Cover the setup with a box.",
      "Saturated (white or coloured) spots carry no information: reduce the exposure.",
      "Then open the Live tab, choose the colour of your light and drag dz until the rings collapse into sharp objects.",
    ],
    docs: "tutorials/your-first-hologram",
  },
  michelson: {
    title: "Michelson interferometer: watch light cancel out",
    intro:
      "A beam splitter sends the laser to two mirrors; both beams come back and overlap. Where they arrive in step they add up, out of step they cancel. A change of the mirror distance by half a wavelength moves the pattern by one fringe - you measure distances of 0.3 µm.",
    parts: [
      "red laser (650 nm) with fibre and fibre holder cube",
      "beam splitter cube",
      "two mirror cubes, at least one kinematic (adjustable)",
      "screen, or the Raspberry Pi camera instead of the screen",
      "1.5 mm hex screwdriver, base plate, puzzle pieces",
    ],
    steps: [
      ["mich_1_fiber.jpg", "Put the fibre holder cube on the plate. Do not switch the laser on yet."],
      ["mich_2_bs.jpg", "Place the beam splitter right after it. It splits the beam into two at 90°."],
      ["mich_3_mirrors.jpg", "Add a mirror at each output of the beam splitter. Check that both mirrors sit at 90° to the beam (open the cube and turn the insert if not)."],
      ["mich_4_screen.jpg", "Put the screen on the fourth side of the beam splitter."],
      ["mich_5_dots.jpg", "Switch the laser on. You will see two dots, one per mirror. Turn the screws of the kinematic mirror, slowly, until the dots lie exactly on top of each other."],
      ["mich_6_rings.jpg", "Add a lens in front of the beam splitter (or replace the screen by the camera). The overlapping spots turn into rings or stripes."],
      ["mich_7_done.jpg", "Done. Touch the table or warm one arm with your hand and watch the fringes move. Count them with the Fringe trace tab."],
    ],
    tips: [
      "Laser safety: never look into the beam or its reflections; keep it below eye level.",
      "One screw at a time, small turns.",
      "With the camera: few wide fringes = mirrors almost parallel. Tilt one mirror a little for straight, evenly spaced stripes.",
    ],
    docs: "tutorials/first-michelson-fringes",
  },
  machzehnder: {
    title: "Mach-Zehnder: the off-axis hologram setup",
    intro:
      "Two separate paths that meet again at a second beam splitter. One path carries the sample (object wave), the other is the reference. Meeting under a small angle they write straight fringes onto the camera; the sample bends them. One image then gives amplitude and phase.",
    parts: [
      "red laser (650 nm) with fibre and fibre holder cube",
      "two beam splitter cubes",
      "two mirror cubes set to 45°, one kinematic",
      "Raspberry Pi camera (or a screen to start)",
      "optional: polarizer to equalise the two arms",
    ],
    steps: [
      ["mz_1_laser.jpg", "Place the fibre holder cube on the plate."],
      ["mz_2_bs1.jpg", "The first beam splitter divides the light into two paths."],
      ["mz_3_mirrors.jpg", "Set both mirrors to 45° and place them so each path turns by 90° towards the far corner."],
      ["mz_4_bs2.jpg", "The second beam splitter brings both paths together. Put the screen behind it."],
      ["mz_5_align.jpg", "Turn the kinematic mirror until both spots overlap on the screen. Make both arms equally bright (block one, then the other; a polarizer helps) for the best contrast."],
      ["mz_6_fringes.jpg", "Fringes appear. Replace the screen by the camera: now you can see fine stripes of a few pixels and align much faster."],
      ["mz_7_done.jpg", "Put a thin transparent sample into ONE arm. In the Live tab the spectrum shows two side peaks: click one (or Find sideband) to get amplitude and phase."],
    ],
    tips: [
      "Aim for 3-5 pixels per fringe. Finer fringes alias, coarser ones push the side peaks into the centre.",
      "Keep everything rigid: every cube on puzzle pieces, on a solid table.",
      "The beam splitters are not polarisation-neutral - rotate the polarizer if the contrast is poor.",
    ],
    docs: "how-to/build-a-mach-zehnder",
  },
};
