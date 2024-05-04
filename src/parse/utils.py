
class ParsingError(Exception):
    pass

def clean_instantiate(class_type: type, *pass_args, **pass_kwargs):
    
    if 'type' in pass_kwargs:
        del pass_kwargs['type']

    try:
        return class_type(*pass_args, **pass_kwargs)
    except Exception as e:
        raise ParsingError(f"Unexpected error when instantiating {class_type}!: \n\t{e}")

def check_kwargs(type_mapping: dict, kwarg_dict: dict, display_name: str) -> None:
    if 'type' not in kwarg_dict:
        raise KeyError(f"{display_name} type not specified!")

    if kwarg_dict['type'] not in type_mapping:
        raise NotImplementedError(f"Invalid {display_name}! Got: `{kwarg_dict['type']}`")
