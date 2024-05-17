
from typing import Any, Mapping, TypeAlias

YamlMap: TypeAlias = dict[str, Any]
YamlList: TypeAlias = list[Any]

class ParsingError(Exception):
    pass

def clean_instantiate(class_type: type, *pass_args: Any, **pass_kwargs: Any):
    
    if 'type' in pass_kwargs:
        del pass_kwargs['type']

    try:
        return class_type(*pass_args, **pass_kwargs)
    except Exception as e:
        raise ParsingError(f"Unexpected error when instantiating {class_type}!: \n\t{e}") from e

def check_kwargs(type_mapping: Mapping[str, type[object]], kwarg_dict: dict[str, Any], display_name: str) -> None:
    if 'type' not in kwarg_dict:
        raise KeyError(f"{display_name} type not specified!")

    if kwarg_dict['type'] not in type_mapping:
        raise NotImplementedError(f"Invalid {display_name}! Got: `{kwarg_dict['type']}`")
