import yaml

from typing import Any

from .utils import YamlMap, YamlList

class Curriculum(yaml.YAMLObject):

    yaml_tag = u'!curriculum'

    def __init__(
            self,
            step_len : int,
            start : int | float,
            stop : int | float,
            step_size : int | float,
        ):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step_size = step_size
        self.step_len = step_len

    @property
    def max_phases(self) -> int:
        return int((self.stop - self.start) / self.step_size)

    @property
    def partitioning_steps(self) -> list[int]:
        return [ self.step_len * i for i in range(1, self.max_phases + 1) ]

    def __repr__(self):
        return f"Curriculum({self.start} to {self.stop}," + \
                f"step_size={self.step_size}, step_length={self.step_len})"

def get_value(
    obj: int | float | YamlList | YamlMap | str | Curriculum, 
    step_num: int
):
    if not isinstance(obj, (Curriculum, dict, list)):
        return obj
    elif isinstance(obj, dict):
        return {
            key : get_value(val, step_num) 
            for key, val in obj.items()
        }
    elif isinstance(obj, list):
        return [ get_value(item, step_num) for item in obj ]

    phases_past = step_num // obj.step_len
    effective_phases = min(phases_past, obj.max_phases)
    result = obj.start + obj.step_size * effective_phases
    casted_result = type(obj.start)(result)
    return casted_result

def get_max_value(
    obj: int | float | Curriculum,
    max_steps: int        
) -> int | float:
    if not isinstance(obj, Curriculum):
        return obj
    return max(
        get_value(obj, max_steps), # pyright: ignore[reportArgumentType]
        get_value(obj, 0) # pyright: ignore[reportArgumentType]
    )

def expand_curriculum(raw_data: YamlMap) -> tuple[list[YamlMap], list[int]]:

    def identify_curriculum_params(data: YamlMap) -> list[list]: # pyright: ignore[reportMissingTypeArgument]
        paths = [ ]
        for key, val in data.items():
            if isinstance(val, Curriculum):
                paths.extend([ [key, val] ])
            elif isinstance(val, dict):
                paths.extend([ [ key ]  + path for path in identify_curriculum_params(val) ] )
        return paths

    curriculums = {
        curriculum : route
        for *route, curriculum in identify_curriculum_params(raw_data)
    }

    partitioning_steps = set([ raw_data['steps'] ])
    currs = list(curriculums.keys())
    for c in currs:
        partitioning_steps = partitioning_steps.union(set(c.partitioning_steps))
    partitioning_steps = sorted(list(partitioning_steps))
    partitioning_steps = filter(lambda step_num: step_num <= raw_data['steps'], partitioning_steps)

    stages = [ ]
    durations = [ ]
    last_boundary: int = 0
    for boundary in partitioning_steps:
        dat = get_value(raw_data, last_boundary)
        duration = boundary - last_boundary
        last_boundary = boundary
        stages.append(dat)
        durations.append(duration)

    return stages, durations
