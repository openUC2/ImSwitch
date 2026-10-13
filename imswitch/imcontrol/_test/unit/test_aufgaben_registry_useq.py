"""Aufgaben task registry (incl. Arkitekt compatibility) and the useq bridge."""
from typing import Annotated

import pytest
import useq
from annotated_types import Gt

from imswitch.imcontrol.model import aufgaben as A
from imswitch.imcontrol.model.aufgaben.hardware import Microscope
from imswitch.imcontrol.model.aufgaben.registry import TaskContext, build_spec, parse_docstring


def test_docstring_args_with_continuations():
    title, body, args = parse_docstring('''Move by

    Moves the stage.

    Args:
        dx_um: Distance in X,
            continued: on the next line.
        speed_um_s (float): Speed.

    Returns:
        nothing
    ''')
    assert title == "Move by" and body == "Moves the stage."
    assert args == {"dx_um": "Distance in X, continued: on the next line.", "speed_um_s": "Speed."}


def test_spec_excludes_injected_arguments_and_reads_units():
    spec = A.REGISTRY["autofocus"]
    assert spec.injected == {"hw": "hw", "ctx": "ctx"}
    props = spec.schema()["properties"]
    assert "hw" not in props and "ctx" not in props
    assert props["range_um"]["x-unit"] == "µm" and props["range_um"]["exclusiveMinimum"] == 0
    assert props["mode"]["enum"] == ["software", "hardware"]


def test_device_choices_resolve_from_the_host():
    schema = A.REGISTRY["set_illumination"].schema({"illumination": lambda: ["LED", "Laser488"]})
    assert schema["properties"]["source"]["enum"] == ["LED", "Laser488"]


def _ok(x_um: float, *, hw: Microscope, ctx: TaskContext) -> None:
    """Ok

    Args:
        x_um: X.
    """


def test_registration_rules():
    def undocumented(x_um: float) -> None:
        """No args section."""
    def positional_hw(x_um: float, hw: Microscope) -> None:
        """P

        Args:
            x_um: X.
        """
    def reserved(value: float) -> None:
        """R

        Args:
            value: V.
        """
    assert build_spec(_ok).injected == {"hw": "hw", "ctx": "ctx"}
    cases = ((undocumented, "no Args"), (positional_hw, "keyword-only"), (reserved, "reserved"))
    for fn, msg in cases:
        with pytest.raises(TypeError, match=msg):
            build_spec(fn)


def test_task_decorator_returns_the_function_unchanged():
    def probe(n: Annotated[int, Gt(0)] = 1) -> int:
        """Probe

        Args:
            n: Count.
        """
        return n
    assert A.task(category="Test", id="_probe")(probe) is probe
    try:
        assert probe(3) == 3 and A.REGISTRY["_probe"].function is probe
    finally:
        A.REGISTRY.pop("_probe", None)


def test_arkitekt_accepts_every_primitive_unchanged():
    define = pytest.importorskip("arkitekt_spec.declare.definition.define")
    registry = pytest.importorskip("arkitekt_spec.declare.structures.registry")
    structures = registry.StructureRegistry()
    for cls, identifier in A.MEMORY_STRUCTURES.items():   # what an app declares once
        structures.register_as_memory_structure(cls, identifier=identifier)
    for task_id, spec in A.REGISTRY.items():
        definition = define.prepare_definition(spec.function, structures, omitkeys=["hw"])
        keys = [a.key for a in definition.args]
        assert keys == list(spec.params_model.model_fields), task_id
        assert "ctx" not in keys  # injected through the task marker


SEQ = useq.MDASequence(
    axis_order="tpgcz",
    stage_positions=[useq.Position(x=0, y=0, z=10, name="a"),
                     useq.Position(x=500, y=0, z=12, name="b")],
    grid_plan=useq.GridRowsColumns(rows=2, columns=2, fov_width=100, fov_height=100),
    z_plan=useq.ZRangeAround(range=4, step=2),
    channels=[useq.Channel(config="LED", exposure=10),
              useq.Channel(config="Laser488", exposure=50)],
    time_plan=useq.TIntervalLoops(interval=1, loops=3),
    autofocus_plan=useq.AxesBasedAF(axes=("p",)),
)


def _events(seq):
    return [(dict(e.index), e.x_pos, e.y_pos, e.z_pos, e.channel and e.channel.config,
             type(e.action).__name__) for e in seq]


@pytest.mark.parametrize("order", ["tpgcz", "tpgzc", "ptgzc", "tpzcg"])
def test_useq_round_trip_gives_identical_events(order):
    seq = SEQ.model_copy(update={"axis_order": tuple(order)})
    job = A.job_from_sequence(seq)
    assert all(n.task != "mda" for n, _ in job.walk())
    assert A.validate(job).ok
    assert _events(A.sequence_from_job(job)) == _events(seq)


def test_unexpandable_sequences_are_kept_whole():
    seq = useq.MDASequence(channels=[useq.Channel(config="LED", do_stack=False)],
                           z_plan=useq.ZRangeAround(range=2, step=1))
    job = A.job_from_sequence(seq)
    assert [n.task for n in job.tasks] == ["mda"]
    assert "do_stack" in job.metadata["kept_whole_because"]
    assert A.validate(job).ok
    assert _events(A.sequence_from_job(job)) == _events(seq)


def test_jobs_with_control_flow_do_not_export():
    with pytest.raises(A.NotExportable):
        A.sequence_from_job(A.examples()["stop_flow_fluidics"])
