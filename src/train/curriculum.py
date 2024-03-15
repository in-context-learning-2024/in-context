import yaml

class Curriculum(yaml.YAMLObject):

    yaml_tag = u'!curriculum'

    def __init__(
            self,
            step_len : int,
            start : int | float,
            stop : int | float,
            step_size : int | float,
        ):
        self.start = start
        self.stop = stop
        self.step_size = step_size
        self.step_len = step_len

    @property
    def max_phases(self) -> int:
        return int((self.stop - self.start) / self.step_size)

    @property
    def partitioning_steps(self):
        return [ self.step_len * i for i in range(1, self.max_phases + 1) ]

    def __repr__(self):
        return f"Curriculum({self.start} to {self.stop}," + \
                f"step_size={self.step_size}, step_length={self.step_len})"


def _get_value(
    obj: int | float | list | dict | str | Curriculum, 
    step_num: int
):
    if not isinstance(obj, (Curriculum, dict, list)):
        return obj
    elif isinstance(obj, dict):
        return {
            key : _get_value(val, step_num) 
            for key, val in obj.items()
        }
    elif isinstance(obj, list):
        return [ _get_value(item, step_num) for item in obj ]

    phases_past = step_num // obj.step_len
    effective_phases = min(phases_past, obj.max_phases)
    result = obj.start + obj.step_size * effective_phases
    casted_result = type(obj.start)(result)
    return casted_result


def expand_curriculum(raw_data: dict, total_steps: int) -> tuple[list[dict], list[int]]:

    def identify_curriculum_params(data: dict) -> list[list]:
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

    partitioning_steps = set([ ])
    currs = list(curriculums.keys())
    for c in currs:
        partitioning_steps = partitioning_steps.union(set(c.partitioning_steps))
    partitioning_steps = sorted(list(partitioning_steps))
    partitioning_steps = filter(lambda step_num: step_num <= total_steps, partitioning_steps)

    stages = [ ]
    durations = [ ]
    last_boundary: int = 0
    for boundary in partitioning_steps:
        dat = _get_value(raw_data, boundary)
        duration = boundary - last_boundary
        last_boundary = boundary
        stages.append(dat)
        durations.append(duration)

    return stages, duration


# filename = "sample.yml"
# with open(filename, 'r') as f:
#     content = f.read()

# d = yaml.load(content, Loader=yaml.Loader)

# fc_dicts, step_counts = expand_curriculum(d['train'], int(1e4))

# import code
# code.interact(local=locals())
