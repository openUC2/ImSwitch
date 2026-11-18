import { mergeSpans, spansFromMarks } from "../components/aufgaben/AufgabenTimeline";

jest.mock("react-plotly.js", () => () => null);

test("marks become spans per step, nested and still running", () => {
  const spans = spansFromMarks([
    [0, "loop", 1], [0.1, "hold", 1], [0.2, "hold", 0], [0.3, "hold", 1], [0.5, "hold", 0], [0.6, "hold", 1],
  ]);
  expect(spans.loop).toEqual([[0, null]]);
  expect(spans.hold).toEqual([[0.1, 0.2], [0.3, 0.5], [0.6, null]]);
  // gaps of 0.1 s merge at gap 0.15 (the bar runs to now) and not at 0.05
  expect(mergeSpans(spans.hold, 1, 0.15, 0, 1)).toEqual([[0.1, 1, 3, true]]);
  expect(mergeSpans(spans.hold, 1, 0.05, 0, 1)).toEqual([[0.1, 0.2, 1, false], [0.3, 0.5, 1, false], [0.6, 1, 1, true]]);
  // only bars inside the visible range [0.55, 1]
  expect(mergeSpans(spans.hold, 1, 0.01, 0.55, 1)).toEqual([[0.6, 1, 1, true]]);
});
