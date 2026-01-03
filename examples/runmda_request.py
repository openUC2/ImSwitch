import requests
from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround, AbsolutePosition

# Create in Jupyter notebook
sequence = MDASequence(
    metadata={"experiment": "xyz_timelapse"},
    stage_positions=[
        AbsolutePosition(x=100.0, y=100.0, z=1.0),
        AbsolutePosition(x=200.0, y=150.0, z=2.0),
        AbsolutePosition(x=150.0, y=200.0, z=3.0)
    ],
    channels=[Channel(config="LED", exposure=10.0), Channel(config="LASER", exposure=5.0)],
    z_plan=ZRangeAround(range=10.0, step=2.0),  # 10µm range
    time_plan=TIntervalLoops(interval=10.0, loops=2),  # 10 timepoints
    axis_order="tpzc"
)

# Send to ImSwitch
response = requests.post(
    "http://100.104.189.88:8001/ExperimentController/run_native_mda_sequence",
    json=sequence.model_dump()
)
print(response.json())
