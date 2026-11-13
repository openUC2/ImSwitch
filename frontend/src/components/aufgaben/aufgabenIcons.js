// src/components/aufgaben/aufgabenIcons.js
// One outlined MUI icon per Aufgaben step, with the category's icon for steps
// added later. Drawn small in text.secondary, next to the step's title.
import AccountTreeOutlined from "@mui/icons-material/AccountTreeOutlined";
import AdjustOutlined from "@mui/icons-material/AdjustOutlined";
import AltRouteOutlined from "@mui/icons-material/AltRouteOutlined";
import BoltOutlined from "@mui/icons-material/BoltOutlined";
import BrightnessMediumOutlined from "@mui/icons-material/BrightnessMediumOutlined";
import CableOutlined from "@mui/icons-material/CableOutlined";
import CallSplitOutlined from "@mui/icons-material/CallSplitOutlined";
import CameraAltOutlined from "@mui/icons-material/CameraAltOutlined";
import CancelOutlined from "@mui/icons-material/CancelOutlined";
import CenterFocusStrongOutlined from "@mui/icons-material/CenterFocusStrongOutlined";
import CodeOutlined from "@mui/icons-material/CodeOutlined";
import ColorLensOutlined from "@mui/icons-material/ColorLensOutlined";
import ControlCameraOutlined from "@mui/icons-material/ControlCameraOutlined";
import CropFreeOutlined from "@mui/icons-material/CropFreeOutlined";
import DataObjectOutlined from "@mui/icons-material/DataObjectOutlined";
import DeviceThermostatOutlined from "@mui/icons-material/DeviceThermostatOutlined";
import DoneAllOutlined from "@mui/icons-material/DoneAllOutlined";
import EjectOutlined from "@mui/icons-material/EjectOutlined";
import ExitToAppOutlined from "@mui/icons-material/ExitToAppOutlined";
import FastForwardOutlined from "@mui/icons-material/FastForwardOutlined";
import FilterCenterFocusOutlined from "@mui/icons-material/FilterCenterFocusOutlined";
import FlagOutlined from "@mui/icons-material/FlagOutlined";
import FlashOffOutlined from "@mui/icons-material/FlashOffOutlined";
import GrainOutlined from "@mui/icons-material/GrainOutlined";
import GridOnOutlined from "@mui/icons-material/GridOnOutlined";
import HelpOutlineOutlined from "@mui/icons-material/HelpOutlineOutlined";
import HomeOutlined from "@mui/icons-material/HomeOutlined";
import HourglassBottomOutlined from "@mui/icons-material/HourglassBottomOutlined";
import HourglassEmptyOutlined from "@mui/icons-material/HourglassEmptyOutlined";
import ImageSearchOutlined from "@mui/icons-material/ImageSearchOutlined";
import InputOutlined from "@mui/icons-material/InputOutlined";
import InsightsOutlined from "@mui/icons-material/InsightsOutlined";
import InvertColorsOffOutlined from "@mui/icons-material/InvertColorsOffOutlined";
import LayersOutlined from "@mui/icons-material/LayersOutlined";
import LightbulbOutlined from "@mui/icons-material/LightbulbOutlined";
import LoopOutlined from "@mui/icons-material/LoopOutlined";
import NearMeOutlined from "@mui/icons-material/NearMeOutlined";
import NotesOutlined from "@mui/icons-material/NotesOutlined";
import OpacityOutlined from "@mui/icons-material/OpacityOutlined";
import OpenWithOutlined from "@mui/icons-material/OpenWithOutlined";
import PhotoCameraOutlined from "@mui/icons-material/PhotoCameraOutlined";
import PlaceOutlined from "@mui/icons-material/PlaceOutlined";
import PublicOutlined from "@mui/icons-material/PublicOutlined";
import RadioButtonUncheckedOutlined from "@mui/icons-material/RadioButtonUncheckedOutlined";
import RepeatOutlined from "@mui/icons-material/RepeatOutlined";
import ReportOutlined from "@mui/icons-material/ReportOutlined";
import ScatterPlotOutlined from "@mui/icons-material/ScatterPlotOutlined";
import SkipNextOutlined from "@mui/icons-material/SkipNextOutlined";
import TerminalOutlined from "@mui/icons-material/TerminalOutlined";
import TerrainOutlined from "@mui/icons-material/TerrainOutlined";
import ThermostatOutlined from "@mui/icons-material/ThermostatOutlined";
import TimerOutlined from "@mui/icons-material/TimerOutlined";
import ToggleOnOutlined from "@mui/icons-material/ToggleOnOutlined";
import TravelExploreOutlined from "@mui/icons-material/TravelExploreOutlined";
import TuneOutlined from "@mui/icons-material/TuneOutlined";
import VerticalAlignCenterOutlined from "@mui/icons-material/VerticalAlignCenterOutlined";
import ViewAgendaOutlined from "@mui/icons-material/ViewAgendaOutlined";
import ViewInArOutlined from "@mui/icons-material/ViewInArOutlined";
import ViewModuleOutlined from "@mui/icons-material/ViewModuleOutlined";
import ViewQuiltOutlined from "@mui/icons-material/ViewQuiltOutlined";
import WaterDropOutlined from "@mui/icons-material/WaterDropOutlined";

const BY_TASK = {
  // loops
  loop_positions: PlaceOutlined,
  loop_wells: GridOnOutlined,
  loop_regions: CropFreeOutlined,
  loop_grid: ViewModuleOutlined,
  zstack: LayersOutlined,
  loop_channels: ColorLensOutlined,
  time_lapse: TimerOutlined,
  repeat: RepeatOutlined,
  // flow
  condition: CallSplitOutlined,
  block: ViewAgendaOutlined,
  phase: FlagOutlined,
  break: ExitToAppOutlined,
  continue: SkipNextOutlined,
  finish: DoneAllOutlined,
  abort: CancelOutlined,
  exit_with_error: ReportOutlined,
  wait: HourglassEmptyOutlined,
  wait_until: HourglassBottomOutlined,
  question: HelpOutlineOutlined,
  set_variable: DataObjectOutlined,
  log: NotesOutlined,
  // acquisition, focus
  capture: PhotoCameraOutlined,
  mda: ViewInArOutlined,
  move_to_item: NearMeOutlined,
  create_focus_map: TerrainOutlined,
  move_to_focus_surface: VerticalAlignCenterOutlined,
  autofocus: CenterFocusStrongOutlined,
  // hardware
  move_to_position: OpenWithOutlined,
  move_relative: ControlCameraOutlined,
  home_axis: HomeOutlined,
  move_to_loading_position: EjectOutlined,
  set_illumination: LightbulbOutlined,
  illumination_off: FlashOffOutlined,
  set_camera: TuneOutlined,
  set_objective: AdjustOutlined,
  fast_stage_scan: FastForwardOutlined,
  stage_prescan: TravelExploreOutlined,
  set_digital_output: ToggleOnOutlined,
  pulse_digital_output: BoltOutlined,
  wait_for_input: InputOutlined,
  pump_volume: OpacityOutlined,
  pump_run: WaterDropOutlined,
  pump_stop: InvertColorsOffOutlined,
  set_valve: AltRouteOutlined,
  set_temperature: ThermostatOutlined,
  read_temperature: DeviceThermostatOutlined,
  http_request: PublicOutlined,
  send_serial: CableOutlined,
  // analysis
  measure_intensity: BrightnessMediumOutlined,
  count_objects: GrainOutlined,
  focus_metric: FilterCenterFocusOutlined,
  detect_background: ImageSearchOutlined,
  points_from_mask: ScatterPlotOutlined,
  regions_from_overview: ViewQuiltOutlined,
  python_script: CodeOutlined,
  external_process: TerminalOutlined,
};

const BY_CATEGORY = {
  Loops: LoopOutlined,
  Flow: AccountTreeOutlined,
  Acquisition: PhotoCameraOutlined,
  Focus: CenterFocusStrongOutlined,
  Stage: OpenWithOutlined,
  Illumination: LightbulbOutlined,
  Camera: CameraAltOutlined,
  Objective: AdjustOutlined,
  "Stage scan": FastForwardOutlined,
  "Digital I/O": ToggleOnOutlined,
  Fluidics: OpacityOutlined,
  Environment: ThermostatOutlined,
  External: PublicOutlined,
  Analysis: InsightsOutlined,
};

export const stepIcon = (task, category) =>
  BY_TASK[task] || BY_CATEGORY[category] || RadioButtonUncheckedOutlined;

// Human words for the capabilities a step can need (catalogue "missing").
const CAPABILITY_NAMES = {
  stage: "a stage",
  camera: "a camera",
  illumination: "illumination",
  led_matrix: "an LED matrix",
  autofocus: "autofocus",
  objective: "an objective turret",
  digital_io: "digital I/O",
  pump: "a pump",
  valve: "a valve",
  temperature: "temperature control",
  sensors: "sensors",
  stage_scan: "firmware stage scans",
  stage_map: "the stage map",
  serial: "a serial device",
};

export const capabilityNames = (missing) =>
  (missing || []).map((cap) => CAPABILITY_NAMES[cap] || cap).join(", ");
