from imswitch.imcommon.model import dirtools
from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger

from typing import Callable, List, Dict, Any, Optional
import threading
import traceback

class WorkflowManager(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

    
    def update(self):
        return None



class WorkflowContext:
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.should_stop = False
        self.current_step_index = 0
        self.event_listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.objects: Dict[str, Any] = {}  # Storage for arbitrary objects
    
    def set_object(self, key: str, obj: Any):
        self.objects[key] = obj
    
    def get_object(self, key: str) -> Any:
        return self.objects.get(key)
    
    def remove_object(self, key: str):
        if key in self.objects:
            del self.objects[key]

    def store_step_result(self, step_id: str, metadata: Dict[str, Any]):
        self.data[step_id] = metadata

    def get_step_result(self, step_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(step_id)

    def update_metadata(self, step_id: str, key: str, value: Any):
        if step_id not in self.data:
            self.data[step_id] = {}
        self.data[step_id][key] = value

    def on(self, event_name: str, callback: Callable[[Dict[str, Any]], None]):
        self.event_listeners.setdefault(event_name, []).append(callback)

    def emit_event(self, event_name: str, payload: Dict[str, Any]):
        for cb in self.event_listeners.get(event_name, []):
            cb(payload)

    def request_stop(self):
        self.should_stop = True


class WorkflowStep:
    def __init__(
        self,
        name: str,
        main_func: Callable[..., Any],
        main_params: Dict[str, Any],
        step_id: str,
        pre_funcs: Optional[List[Callable[..., Any]]] = None,
        pre_params: Optional[Dict[str, Any]] = None,
        post_funcs: Optional[List[Callable[..., Any]]] = None,
        post_params: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.main_func = main_func
        self.main_params = main_params
        self.step_id = step_id
        self.pre_funcs = pre_funcs or []
        self.pre_params = pre_params or {}
        self.post_funcs = post_funcs or []
        self.post_params = post_params or {}
        # Allowing retries handling (if provided in main_params)
        self.max_retries = self.main_params.pop("max_retries", 0)

    def run(self, context: WorkflowContext):
        if context.should_stop:
            return None  # Don't run if stop requested

        # Merge params for pre and post functions
        # We'll pass metadata and context, plus pre/post_params
        # This lets pre/post funcs accept flexible arguments
        metadata = {
            "step_id": self.step_id,
            **self.main_params,
        }

        # Emit event before step starts
        context.emit_event("progress", {"status": "started", "step_id": self.step_id, "name": self.name})

        # Run pre-processing functions
        metadata["pre_result"] = []
        for f in self.pre_funcs:
            # Merge context, metadata and pre_params
            merged_pre_params = {**self.pre_params, "context": context, "metadata": metadata}
            result = f(**merged_pre_params)
            metadata["pre_result"].append(result)
            if context.should_stop:
                return None

        # Run main function with error handling and retries
        retries = self.max_retries
        while True:
            try:
                result = self.main_func(**self.main_params)
                metadata["result"] = result
                break
            except Exception as e:
                print(e)
                metadata["error"] = str(e)
                metadata["traceback"] = traceback.format_exc()
                if retries > 0:
                    retries -= 1
                    # Optionally emit event about retry
                    context.emit_event("progress", {"status": "retrying", "step_id": self.step_id})
                else:
                    # No more retries, stop workflow or handle gracefully
                    context.should_stop = True
                    context.store_step_result(self.step_id, metadata)
                    context.emit_event("progress", {"status": "failed", "step_id": self.step_id})
                    return None

        # Run post-processing functions
        metadata["post_result"] = []
        for f in self.post_funcs:
            merged_post_params = {**self.post_params, "context": context, "metadata": metadata}
            result = f(**merged_post_params)
            metadata["post_result"].append(result)
            if context.should_stop:
                return None

        # Store final metadata in the context
        context.store_step_result(self.step_id, metadata)

        # Emit event that step completed
        context.emit_event("progress", {"status": "completed", "step_id": self.step_id, "name": self.name})
        return metadata["result"]

class Workflow:
    def __init__(self, steps: List[WorkflowStep]):
        self.steps = steps

    def run(self, context: Optional[WorkflowContext] = None):
        # Either use the given context or create a new one
        context = context or WorkflowContext()

        # Resume from current_step_index if previously stopped
        for i in range(context.current_step_index, len(self.steps)):
            step = self.steps[i]
            if context.should_stop:
                break
            step.run(context)
            context.current_step_index = i + 1  # Update progress for resume

        return context

    def run_in_background(self, context: Optional[WorkflowContext] = None):
        context = context or WorkflowContext()
        def background_run():
            try:
                self.run(context)
            finally:
                # Mark workflow as finished
                WorkflowsManager.instance().workflow_finished()

        t = threading.Thread(target=background_run)
        t.start()
        return context, t
    
    


########################################
# A Global WorkflowsManager to handle states
########################################

class WorkflowsManager:
    _instance = None

    @staticmethod
    def instance():
        if WorkflowsManager._instance is None:
            WorkflowsManager._instance = WorkflowsManager()
        return WorkflowsManager._instance

    def __init__(self):
        self.current_workflow = None
        self.current_context = None
        self.current_thread = None

    def start_workflow(self, wf: Workflow, context: WorkflowContext):
        if self.current_workflow is not None:
            raise RuntimeError("A workflow is already running. Stop or wait before starting another one.")
        self.current_workflow = wf
        self.current_context, self.current_thread = wf.run_in_background(context)
        return {"status": "started"}

    def pause_workflow(self):
        if self.current_context is None:
            return {"status": "no_workflow"}
        self.current_context.request_pause()
        return {"status": "paused"}

    def resume_workflow(self):
        if self.current_context is None:
            return {"status": "no_workflow"}
        self.current_context.request_resume()
        return {"status": "resumed"}

    def stop_workflow(self):
        if self.current_context is None:
            return {"status": "no_workflow"}
        self.current_context.request_stop()
        return {"status": "stopping"}

    def workflow_finished(self):
        self.current_workflow = None
        self.current_context = None
        self.current_thread = None

    def get_status(self):
        if self.current_context is None:
            return {"status": "idle"}
        if self.current_context.should_stop:
            return {"status": "stopping"}
        if self.current_context.should_pause:
            return {"status": "paused"}
        return {"status": "running"}


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
