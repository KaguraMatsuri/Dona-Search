_REGISTRY = {}

def register(category: str, name: str):
    def decorator(cls):
        key = f"{category}:{name}"
        _REGISTRY[key] = cls
        return cls
    return decorator

def get(category: str, name: str):
    key = f"{category}:{name}"
    if key not in _REGISTRY:
        raise ValueError(f"Unknown {category}: {name}. Available: {list_names(category)}")
    return _REGISTRY[key]

def list_names(category: str) -> list[str]:
    prefix = f"{category}:"
    return [k.split(":", 1)[1] for k in _REGISTRY if k.startswith(prefix)]
