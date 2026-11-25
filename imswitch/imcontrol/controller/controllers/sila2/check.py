"""Exercise every SiLA 2 endpoint of an ImSwitch microscope, as a workcell would.

    imswitch-sila-check --host 192.168.1.42 --port 50052
    imswitch-sila-check --discover --motion --light      # virtual microscope: everything

Connects like the UniteLabs client tutorial (``sila.client.Client``), reads every property,
takes one value from every observable property, calls every command of the ImSwitch features
and checks the defined errors of the refusals. Steps that move the stage need ``--motion``,
steps that switch light on need ``--light``; ``--home`` adds homing (end switches). Every
command and property of the ImSwitch features is either run or listed as skipped. Exit code 1
when a step fails or an element is neither run nor skipped.

On a real instrument: run without --motion/--light first, and with them only when the stage is
free and nobody looks into the light path.
"""

import argparse
import asyncio
import json
import sys
import time

from sila.client import Client, ClientConfig

FEATURES = ("StageController", "ImagingController", "ExperimentController",
            "LabwareTransferSiteController")
# Position Index is a one-field structure; on the wire it is the bare index.
HANDOVER = {"Position": "Stage", "SubPosition": 1}
INTERNAL = 1


class Check:
    """Runs steps against one client and records the outcome of each."""

    def __init__(self, client, motion: bool, light: bool, home: bool, intensity: float):
        self.client = client
        # feature identifier (e.g. "StageController") -> feature
        self.features = {fid.split("/")[-2]: feat for fid, feat in client.features.items()}
        self.allowed = {"motion": motion, "light": light, "home": home}
        self.intensity = intensity
        self.results = []  # (element, status, detail)
        self.touched = set()

    def record(self, element, status, detail=""):
        self.results.append((element, status, str(detail)[:200]))
        self.touched.add(element)
        mark = {"ok": "ok  ", "fail": "FAIL", "skip": "skip"}[status]
        print(f"[{mark}] {element:<55} {detail}"[:160], flush=True)

    async def call(self, feature, command, *, expect_error="", needs=(), timeout=120, **params):
        """Run one command; returns its responses, or None when skipped or failed."""
        element = f"{feature}.{command}"
        missing = [n for n in needs if not self.allowed[n]]
        if missing:
            self.record(element, "skip", f"needs --{' --'.join(missing)}")
            return None
        handle = self.features[feature].commands[command]
        try:
            responses = await self._execute(handle, params, timeout)
        except Exception as e:  # a defined error is the expected outcome of some steps
            name = getattr(e, "identifier", "") or type(e).__name__
            ok = bool(expect_error) and name == expect_error
            self.record(element, "ok" if ok else "fail", f"{name}: {e}")
            return None
        if expect_error:
            self.record(element, "fail", f"expected {expect_error}, got {dict(responses)}")
            return None
        self.record(element, "ok", _short(responses))
        return responses

    async def _execute(self, handle, params, timeout):
        if type(handle).__name__ != "ObservableCommand":
            return await handle.execute(parameters=params)
        execution = await handle.execute(parameters=params)
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
    await c.call(f, "MoveToLoadingPosition", needs=("motion",))
    await c.call(f, "Home", needs=("motion", "home"), timeout=300)


async def _plate(c: Check, source: str):
    f = "ExperimentController"
    labware = await c.features[f].properties["Labware"].read()
    plate = next((n for n in labware if "96" in n), labware[0] if labware else "")
    await c.call(f, "SetPlateOrigin", Labware=plate, Well="Z99", expect_error="UnknownDevice")
    await c.call(f, "SetPlateOrigin", Labware=plate, Well="A1")
    await c.call(f, "MoveToWell", Well="A2", needs=("motion",))
    await c.call(f, "MoveToWell", Well="A1", needs=("motion",))
    await c.call(f, "ImageWells", Wells=["A1", "A2"], Illumination=source,
                 Intensity=c.intensity, ExposureTime=20.0, Autofocus=False,
                 needs=("motion", "light"), timeout=600)
    await c.call(f, "StartExperiment", Experiment="{}", expect_error="ValidationError")
    for command in ("PauseExperiment", "ResumeExperiment", "StopExperiment"):
        await c.call(f, command)


async def _handover(c: Check):
    f = "LabwareTransferSiteController"
    await c.call(f, "LabwareDelivered", HandoverPosition=HANDOVER,
                 expect_error="InvalidCommandSequence")
    await c.call(f, "PrepareForInput", HandoverPosition={**HANDOVER, "Position": "Nowhere"},
                 InternalPosition=INTERNAL, LabwareType="plate", LabwareUniqueID="x",
                 expect_error="InvalidCommandSequence")
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


async def run_check(client, motion=False, light=False, home=False, intensity=0.0) -> Check:
    """Run every step; returns the Check with its results."""
    c = Check(client, motion, light, home, intensity)
    await _service(c)
    for feature in FEATURES:
        if feature in c.features:
            await c.read_properties(feature)
    source = await _imaging(c)
    await _stage(c)
    await _plate(c, source)
    await _handover(c)
    _untouched(c)
    return c


def _untouched(c: Check) -> None:
    for feature in FEATURES:
        feat = c.features.get(feature)
        for name in list(feat.commands) + list(feat.properties) if feat else []:
            if f"{feature}.{name}" not in c.touched:
                c.record(f"{feature}.{name}", "fail", "not exercised by the check")


async def _connect(args) -> Client:
    if args.discover:
        return await Client.discover(uuid=args.uuid or None, name=args.name or None,
                                     timeout=args.discover_timeout * 1000)
    root = open(args.ca, "rb").read() if args.ca else None
    return Client(ClientConfig(hostname=args.host, port=args.port, tls=args.tls,
                               root_certificates=root, default_timeout=60_000))


async def _main(args) -> int:
    client = await _connect(args)
    async with client:
        c = await run_check(client, args.motion, args.light, args.home, args.intensity)
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
    p.add_argument("--discover", action="store_true", help="find the server over mDNS")
    p.add_argument("--uuid", help="with --discover: the server UUID")
    p.add_argument("--name", help="with --discover: the server name")
    p.add_argument("--discover-timeout", type=float, default=10.0, help="seconds")
    p.add_argument("--motion", action="store_true", help="also run steps that move the stage")
    p.add_argument("--light", action="store_true", help="also run steps that switch light on")
    p.add_argument("--home", action="store_true", help="also home the stage (end switches)")
    p.add_argument("--intensity", type=float, default=0.0,
                   help="illumination intensity for the light steps (units of the source)")
    p.add_argument("--json", help="write the results to this file")
    return asyncio.run(_main(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
