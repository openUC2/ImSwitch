"""Exercise every SiLA 2 endpoint of an ImSwitch microscope, as a workcell would.

    imswitch-sila-check --host 192.168.1.42 --port 50052
    imswitch-sila-check --discover --motion --light      # virtual microscope: everything

Needs only ``unitelabs-sila``, not a running ImSwitch: it can run on the workcell's computer.
``--list`` shows the SiLA servers announcing themselves on the network.

Connects like the UniteLabs client tutorial (``sila.client.Client``), reads every property,
takes one value from every observable property, calls every command of the ImSwitch features
and checks the defined errors of the refusals. Steps that move the stage need ``--motion``,
steps that switch light on need ``--light``; ``--home`` adds homing (end switches) and
``--teach`` the steps that overwrite saved settings (plate origin, loading position, the
server-initiated connection mode, which is set back). Every command and property of the
server's features is either run or listed as skipped. Exit code 1 when a step fails or an
element is neither run nor skipped.

On a real instrument: run without flags first; add --motion/--light only when the stage is
free and nobody looks into the light path, and --teach only on purpose.
"""

import argparse
import asyncio
import json
import sys
import time

from sila.client import Client, ClientConfig

FEATURES = ("StageController", "ImagingController", "ExperimentController", "JobController",
            "LabwareTransferSiteController", "LockController")
CORE = ("SiLAService", "ConnectionConfigurationService")  # every CDK server has these
# Position Index is a one-field structure; on the wire it is the bare index.
HANDOVER = {"Position": "Stage", "SubPosition": 1}
INTERNAL = 1


class Check:
    """Runs steps against one client and records the outcome of each."""

    def __init__(self, client, allowed: dict, intensity: float, job: str):
        self.client = client
        # feature identifier (e.g. "StageController") -> feature
        self.features = {fid.split("/")[-2]: feat for fid, feat in client.features.items()}
        self.allowed = allowed  # {"motion", "light", "home", "teach"} -> bool
        self.intensity = intensity
        self.job = job
        self.results = []  # (element, status, detail)
        self.touched = set()

    def record(self, element, status, detail=""):
        self.results.append((element, status, str(detail)[:200]))
        self.touched.add(element)
        mark = {"ok": "ok  ", "fail": "FAIL", "skip": "skip"}[status]
        print(f"[{mark}] {element:<55} {detail}"[:160], flush=True)

    async def call(self, feature, command, *, expect_error="", needs=(), timeout=120,
                   metadata=None, quiet=False, **params):
        """Run one command; returns its responses, or None when skipped or failed."""
        element = f"{feature}.{command}"
        missing = [n for n in needs if not self.allowed[n]]
        if missing:
            self.record(element, "skip", f"needs --{' --'.join(missing)}")
            return None
        handle = self.features[feature].commands[command]
        try:
            responses = await self._execute(handle, params, timeout, metadata)
        except Exception as e:  # a defined error is the expected outcome of some steps
            name = getattr(e, "identifier", "") or type(e).__name__
            ok = bool(expect_error) and name == expect_error
            self.record(element, "ok" if ok else "fail", f"{name}: {e}")
            return None
        if expect_error:
            self.record(element, "fail", f"expected {expect_error}, got {dict(responses)}")
            return None
        if not quiet:
            self.record(element, "ok", _short(responses))
        return responses

    async def start(self, feature, command, **params):
        """Start an observable command and return its execution without waiting."""
        return await self.features[feature].commands[command].execute(parameters=params)

    async def _execute(self, handle, params, timeout, metadata):
        if type(handle).__name__ != "ObservableCommand":
            return await handle.execute(parameters=params, metadata=metadata)
        execution = await handle.execute(parameters=params, metadata=metadata)
        deadline = time.monotonic() + timeout
        while not execution.done:
            if time.monotonic() > deadline:
                raise TimeoutError(f"not done after {timeout} s")
            await asyncio.sleep(0.2)
        return await execution.get_responses()

    async def read_properties(self, feature):
        for name, prop in self.features[feature].properties.items():
            element = f"{feature}.{name}"
            try:
                if hasattr(prop, "read"):
                    value = await prop.read()
                else:
                    stream = await prop.subscribe()
                    try:
                        value = await asyncio.wait_for(anext(stream), timeout=5)
                    finally:
                        stream.cancel() if hasattr(stream, "cancel") else stream.close()
                self.record(element, "ok", _short(value))
            except Exception as e:
                self.record(element, "fail", f"{type(e).__name__}: {e}")


def _short(value) -> str:
    if isinstance(value, dict):
        return json.dumps({k: (f"<{len(v)} bytes>" if isinstance(v, bytes) else v)
                           for k, v in value.items()}, default=str)[:120]
    return str(value)[:120]


async def _imaging(c: Check):
    f = "ImagingController"
    snap = await c.call(f, "SnapImage", Detector="", MaxSize=256)
    if snap is not None and not bytes(snap["Image"]).startswith(b"\xff\xd8"):
        c.record(f"{f}.SnapImage", "fail", "the image is not a JPEG")
    await c.call(f, "SnapImage", Detector="no-such-camera", MaxSize=0,
                 expect_error="UnknownDevice")
    await c.call(f, "SetExposureTime", Detector="", ExposureTime=20.0)
    await c.call(f, "SaveImage", Detector="", Name="silacheck")
    sources = await c.features[f].properties["IlluminationSources"].read()
    source = sources[0] if sources else ""
    await c.call(f, "SetIllumination", Source=source, Intensity=c.intensity, Enabled=True,
                 needs=("light",))
    await c.call(f, "SwitchOffIllumination")
    await c.call(f, "Autofocus", Range=20.0, Step=5.0, needs=("motion", "light"))
    current = await c.features[f].properties["CurrentObjective"].read()
    await c.call(f, "MoveToObjective", Slot=current, needs=("motion",))  # stays in place
    await c.call(f, "MoveToObjective", Slot=99, expect_error="UnknownDevice")
    return source


async def _stage(c: Check):
    f = "StageController"
    await c.call(f, "MoveBy", X=0.0, Y=0.0, Z=0.0)  # no-op, no motion
    pos = await c.call(f, "MoveBy", X=50.0, Y=0.0, Z=0.0, needs=("motion",))
    if pos is not None:
        await c.call(f, "MoveBy", X=-50.0, Y=0.0, Z=0.0, needs=("motion",))
        p = pos["Position"]
        await c.call(f, "MoveTo", X=p["X"] - 50.0, Y=p["Y"], needs=("motion",))
        await c.call(f, "MoveFocusTo", Z=p["Z"], needs=("motion",))
    else:
        c.record(f"{f}.MoveTo", "skip", "needs --motion")
        c.record(f"{f}.MoveFocusTo", "skip", "needs --motion")
    await c.call(f, "Stop")
    await c.call(f, "TeachLoadingPosition", needs=("teach",))
    await c.call(f, "MoveToLoadingPosition", needs=("motion",))
    await c.call(f, "Home", needs=("motion", "home"), timeout=300)


async def _plate(c: Check, source: str):
    f = "ExperimentController"
    await c.call(f, "StartExperiment", Experiment="{}", expect_error="ValidationError")
    for command in ("PauseExperiment", "ResumeExperiment", "StopExperiment"):
        await c.call(f, command)
    props = c.features[f].properties
    labware = await props["Labware"].read()
    plate = next((n for n in labware if "96" in n), labware[0] if labware else "")
    await c.call(f, "SetPlateOrigin", Labware=plate, Well="Z99", expect_error="UnknownDevice")
    await c.call(f, "SetPlateOrigin", Labware=plate, Well="A1", needs=("teach",))
    calibrated = await props["CalibratedLabware"].read()
    other = next((n for n in labware if n not in calibrated), "")
    if other:  # refused before anything moves
        await c.call(f, "MoveToWell", Labware=other, Well="A1", expect_error="PlateNotCalibrated")
    if plate not in calibrated:
        plate = calibrated[0] if calibrated else ""
    if not plate:
        for command in ("MoveToWell", "ImageWells"):
            c.record(f"{f}.{command}", "skip", "no taught plate origin (--teach teaches one)")
        return
    await c.call(f, "ImageWells", Labware=plate, Wells=[], Illumination=source,
                 Intensity=c.intensity, ExposureTime=20.0, Autofocus=False,
                 expect_error="ValidationError")
    await c.call(f, "MoveToWell", Labware=plate, Well="A2", needs=("motion",))
    await c.call(f, "MoveToWell", Labware=plate, Well="A1", needs=("motion",))
    await c.call(f, "ImageWells", Labware=plate, Wells=["A1", "A2"], Illumination=source,
                 Intensity=c.intensity, ExposureTime=20.0, Autofocus=False,
                 needs=("motion", "light"), timeout=600)
    if c.allowed["motion"] and c.allowed["light"]:
        await _while_busy(c, plate, source)


async def _while_busy(c: Check, plate: str, source: str):
    """While a scan runs, moves, autofocus and a second scan are refused as InstrumentBusy."""
    f = "ExperimentController"
    run = await c.start(f, "ImageWells", Labware=plate, Wells=["A1:A6"], Illumination=source,
                        Intensity=c.intensity, ExposureTime=20.0, Autofocus=False)
    await asyncio.sleep(1.0)
    await c.call("StageController", "MoveBy", X=10.0, Y=0.0, Z=0.0,
                 expect_error="InstrumentBusy")
    await c.call("ImagingController", "Autofocus", Range=20.0, Step=5.0,
                 expect_error="InstrumentBusy")
    await c.call(f, "ImageWells", Labware=plate, Wells=["A1"], Illumination=source,
                 Intensity=c.intensity, ExposureTime=20.0, Autofocus=False,
                 expect_error="InstrumentBusy")
    while not run.done:
        await asyncio.sleep(0.5)
    await run.get_responses()


async def _jobs(c: Check):
    f = "JobController"
    jobs = await c.features[f].properties["Jobs"].read()
    await c.call(f, "RunJob", Name="no-such-job", Parameters="", Confirmed=False,
                 expect_error="UnknownDevice")
    await c.call(f, "RunJob", Name=c.job, Parameters="not json", Confirmed=False,
                 expect_error="ValidationError")
    if c.job in jobs:
        await c.call(f, "RunJob", Name=c.job, Parameters="", Confirmed=False,
                     needs=("motion", "light"), timeout=600)
    else:
        c.record(f"{f}.RunJob", "skip", f"no job '{c.job}' on the microscope (--job NAME)")
    await c.call(f, "AbortJob")


async def _lock(c: Check):
    f = "LockController"
    lock = await c.client.LockController.LockIdentifier.from_native(c.client, "silacheck")
    await c.call(f, "LockServer", LockIdentifier="silacheck", Timeout=60)
    await c.call(f, "LockServer", LockIdentifier="other", Timeout=60,
                 expect_error="ServerAlreadyLocked")
    await c.call("StageController", "Stop", metadata=[lock])  # the holder goes on working
    await c.call(f, "UnlockServer", LockIdentifier="wrong", expect_error="InvalidLockIdentifier")
    await c.call(f, "UnlockServer", LockIdentifier="silacheck")
    await c.call(f, "UnlockServer", LockIdentifier="silacheck", expect_error="ServerNotLocked")
    await c.call(f, "LockServer", LockIdentifier="forever", Timeout=0)  # 0 s = no timeout
    await asyncio.sleep(1.0)
    await c.call(f, "UnlockServer", LockIdentifier="forever")  # fails if it had expired
    await _lock_expiry(c)


async def _lock_expiry(c: Check):
    """A lock that times out is reported to IsLocked subscribers without anyone asking."""
    f = "LockController"
    await c.call(f, "LockServer", LockIdentifier="short", Timeout=1, quiet=True)
    stream = await c.features[f].properties["IsLocked"].subscribe()
    try:
        seen = []
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and seen[-1:] != [False]:
            seen.append(await asyncio.wait_for(anext(stream), timeout=4.0))
    except asyncio.TimeoutError:
        pass
    finally:
        stream.cancel() if hasattr(stream, "cancel") else stream.close()
    c.record(f"{f}.IsLocked", "ok" if seen[-1:] == [False] else "fail",
             f"a 1 s lock expired, subscribers saw {seen}")


async def _handover(c: Check):
    f = "LabwareTransferSiteController"
    await c.call(f, "LabwareDelivered", HandoverPosition=HANDOVER,
                 expect_error="InvalidCommandSequence")
    await c.call(f, "PrepareForInput", HandoverPosition={**HANDOVER, "Position": "Nowhere"},
                 InternalPosition=INTERNAL, LabwareType="plate", LabwareUniqueID="x",
                 expect_error="ValidationError")
    await c.call(f, "PrepareForInput", HandoverPosition=HANDOVER, InternalPosition=2,
                 LabwareType="plate", LabwareUniqueID="x", expect_error="ValidationError")
    ok = await c.call(f, "PrepareForInput", HandoverPosition=HANDOVER, InternalPosition=INTERNAL,
                      LabwareType="96 well plate", LabwareUniqueID="silacheck-1",
                      needs=("motion",))
    if ok is None:
        reason = "needs --motion" if not c.allowed["motion"] else "PrepareForInput failed"
        for command in ("LabwareDelivered", "PrepareForOutput", "LabwareRemoved"):
            c.record(f"{f}.{command}", "skip" if not c.allowed["motion"] else "fail", reason)
        return
    await c.call(f, "LabwareDelivered", HandoverPosition=HANDOVER)
    await c.call(f, "PrepareForInput", HandoverPosition=HANDOVER, InternalPosition=INTERNAL,
                 LabwareType="96 well plate", LabwareUniqueID="silacheck-2",
                 expect_error="InvalidCommandSequence")  # a plate is already on the stage
    await c.call(f, "PrepareForOutput", HandoverPosition=HANDOVER, InternalPosition=INTERNAL)
    await c.call(f, "LabwareRemoved", HandoverPosition=HANDOVER)


async def _service(c: Check) -> str:
    service = c.features["SiLAService"].properties
    info = {name: await service[name].read() for name in
            ("ServerName", "ServerType", "ServerUUID", "ServerVersion")}
    print(f"Server: {info}", flush=True)
    missing = [f for f in FEATURES if f not in c.features]
    for name in missing:
        c.record(name, "fail", "feature not implemented by the server")
    return info["ServerUUID"]


async def _core(c: Check):
    """The SiLA Service (every feature definition must come back as XML) and the
    connection configuration a cloud or server-initiated setup uses."""
    f = "SiLAService"
    implemented = await c.features[f].properties["ImplementedFeatures"].read()
    bad = []
    for fid in implemented:
        xml = await c.call(f, "GetFeatureDefinition", FeatureIdentifier=fid, quiet=True)
        if xml is None or "<Feature" not in xml["FeatureDefinition"]:
            bad.append(fid)
    c.record(f"{f}.GetFeatureDefinition", "fail" if bad else "ok",
             f"no definition for {bad}" if bad else f"{len(implemented)} feature definitions")
    await c.call(f, "GetFeatureDefinition", FeatureIdentifier="org.example/x/Nope/v1",
                 expect_error="UnimplementedFeature")
    name = await c.features[f].properties["ServerName"].read()
    await c.call(f, "SetServerName", ServerName=name)  # the same name: nothing changes
    f = "ConnectionConfigurationService"
    await c.call(f, "DisconnectSiLAClient", ClientName="silacheck-none", Remove=False,
                 expect_error="InvalidSiLAClient")
    # Toggle the server-initiated mode and set it back: it is saved on the microscope, and a
    # workcell's cloud connection drops while it is off.
    on = await c.features[f].properties["ServerInitiatedConnectionModeStatus"].read()
    toggle = ["EnableServerInitiatedConnectionMode", "DisableServerInitiatedConnectionMode"]
    for command in (toggle[::-1] if on else toggle):
        await c.call(f, command, needs=("teach",))
    c.record(f"{f}.ConnectSiLAClient", "skip", "connects the server out to a client; per site")


async def run_check(client, motion=False, light=False, home=False, teach=False, intensity=0.0,
                    job="autofocus_here_macro") -> Check:
    """Run every step; returns the Check with its results."""
    allowed = {"motion": motion, "light": light, "home": home, "teach": teach}
    c = Check(client, allowed, intensity, job)
    await _service(c)
    for feature in CORE + FEATURES:
        if feature in c.features:
            await c.read_properties(feature)
    await _core(c)
    source = await _imaging(c)
    await _stage(c)
    await _plate(c, source)
    await _jobs(c)
    await _handover(c)
    await _lock(c)
    _untouched(c)
    return c


def _untouched(c: Check) -> None:
    for feature in CORE + FEATURES:
        feat = c.features.get(feature)
        for name in list(feat.commands) + list(feat.properties) if feat else []:
            if f"{feature}.{name}" not in c.touched:
                c.record(f"{feature}.{name}", "fail", "not exercised by the check")


async def _discover(args) -> tuple:
    """Host and port of the server found over mDNS (by --name / --uuid, else the only one)."""
    from sila.client.discovery import Discovery
    async with Discovery() as discovery:
        if args.uuid or args.name:
            service = await discovery.wait_for(uuid=args.uuid or None, name=args.name or None,
                                               timeout=args.discover_timeout * 1000)
        else:
            await asyncio.sleep(args.discover_timeout)
            services = list(discovery.services.values())
            if len(services) != 1:
                raise SystemExit(f"{len(services)} SiLA servers found; pick one with --name "
                                 "or --uuid (see --list).")
            service = services[0]
    print(f"Found {service.name} ({service.uuid}) at {service.addresses[0]}:{service.port}")
    return service.addresses[0], service.port


async def _connect(args) -> Client:
    # Not Client.discover: in unitelabs-sila 0.14 it drops TLS settings and the timeout.
    host, port = await _discover(args) if args.discover else (args.host, args.port)
    root = open(args.ca, "rb").read() if args.ca else None
    return Client(ClientConfig(hostname=host, port=port, tls=args.tls,
                               root_certificates=root, default_timeout=60_000))


async def _list(seconds: float) -> int:
    """Print the SiLA servers that announce themselves on the network."""
    from sila.client.discovery import Discovery
    async with Discovery() as discovery:
        await asyncio.sleep(seconds)
        services = list(discovery.services.values())
    for s in services:
        print(f"{s.name}  uuid={s.uuid}  {', '.join(s.addresses)}:{s.port}")
    if not services:
        print("No SiLA server found (multicast blocked? try --host/--port).")
    return 0 if services else 1


async def _main(args) -> int:
    if args.list:
        return await _list(args.discover_timeout)
    client = await _connect(args)
    async with client:
        c = await run_check(client, args.motion, args.light, args.home, args.teach,
                            args.intensity, args.job)
    failed = [r for r in c.results if r[1] == "fail"]
    counts = {s: sum(1 for r in c.results if r[1] == s) for s in ("ok", "fail", "skip")}
    print(f"\n{counts['ok']} ok, {counts['fail']} failed, {counts['skip']} skipped")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump([dict(zip(("element", "status", "detail"), r)) for r in c.results], fh,
                      indent=1)
    return 1 if failed else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=50052)
    p.add_argument("--tls", action="store_true", help="connect with TLS")
    p.add_argument("--ca", help="PEM file of the CA that signed the server certificate")
    p.add_argument("--list", action="store_true",
                   help="only list the SiLA servers found over mDNS and exit")
    p.add_argument("--discover", action="store_true", help="find the server over mDNS")
    p.add_argument("--uuid", help="with --discover: the server UUID")
    p.add_argument("--name", help="with --discover: the server name")
    p.add_argument("--discover-timeout", type=float, default=10.0, help="seconds")
    p.add_argument("--motion", action="store_true", help="also run steps that move the stage")
    p.add_argument("--light", action="store_true", help="also run steps that switch light on")
    p.add_argument("--home", action="store_true", help="also home the stage (end switches)")
    p.add_argument("--teach", action="store_true",
                   help="also overwrite the taught plate origin and loading position")
    p.add_argument("--job", default="autofocus_here_macro",
                   help="saved job RunJob runs with --motion --light")
    p.add_argument("--intensity", type=float, default=0.0,
                   help="illumination intensity for the light steps (units of the source)")
    p.add_argument("--json", help="write the results to this file")
    return asyncio.run(_main(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
