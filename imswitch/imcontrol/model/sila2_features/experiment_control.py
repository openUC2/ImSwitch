"""
SiLA2 Experiment Control Feature for OpenUC2 ImSwitch.

Provides SiLA2 commands for running automated experiments
via the ExperimentController. Experiments are described as a JSON payload
matching the Experiment Pydantic model used by the ExperimentController.
"""

import abc

from ._compat import (
    SilaFeatureBase,
    UnobservableProperty,
    UnobservableCommand,
    Response,
)


class ExperimentControlFeature(SilaFeatureBase, metaclass=abc.ABCMeta):
    """
    Experiment Control Feature

    Provides commands to submit and monitor automated experiments
    on the OpenUC2 microscope. Experiments are described using the
    ImSwitch Experiment JSON schema and executed by the ExperimentController.
    """

    def __init__(self):
        super().__init__(
            originator="org.openuc2",
            category="microscopy",
            version="1.0",
            maturity_level="Draft",
        )

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="ExperimentId")
    async def start_experiment(self, experiment_json: str) -> str:
        """
        Start an automated experiment.

        Accepts a JSON string matching the ImSwitch Experiment schema.
        The experiment is handed to the ExperimentController for execution.

        .. parameter:: experiment_json: JSON string describing the experiment
            (see ``Experiment`` model in ImSwitch ExperimentController).
        .. return:: An identifier for the experiment run.
        """

    @abc.abstractmethod
    @UnobservableProperty()
    async def get_experiment_status(self) -> str:
        """
        Get the current status of the running experiment.

        .. return:: JSON string with status information.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="Result")
    async def pause_experiment(self) -> bool:
        """
        Pause the currently running experiment.

        .. return:: True if pause succeeded, False otherwise.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="Result")
    async def resume_experiment(self) -> bool:
        """
        Resume a paused experiment.

        .. return:: True if resume succeeded, False otherwise.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="Result")
    async def stop_experiment(self) -> bool:
        """
        Stop / abort the currently running experiment.

        .. return:: True if stop succeeded, False otherwise.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="SchemaJson")
    async def get_experiment_schema(self) -> str:
        """
        Return the JSON schema describing the Experiment data model.

        Clients can use this to build valid experiment payloads.

        .. return:: JSON schema string for the Experiment model.
        """
