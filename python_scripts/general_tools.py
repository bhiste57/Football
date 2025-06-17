class InputProperty:
    def __init__(self, unit, description, default=None, required=False):
        self.unit = unit
        self.description = description
        self.default = default
        self.required = required
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, self.default)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)
   
        
class MagicProperty:
    def __init__(self, overwritable=False, cached=False):
        self.overwritable = overwritable
        self.cached = cached

    def __call__(self, func):
        name = func.__name__
        private_name = f"__magic_{name}"

        class Descriptor:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                if self.overwritable and hasattr(obj, private_name):
                    return getattr(obj, private_name)
                if self.cached:
                    if not hasattr(obj, private_name):
                        value = func(obj)
                        setattr(obj, private_name, value)
                    return getattr(obj, private_name)
                return func(obj)

            def __set__(self, obj, value):
                if not (self.overwritable or self.cached):
                    raise AttributeError(f"Can't set attribute '{name}'")
                setattr(obj, private_name, value)

            def __set_name__(self, owner, attr_name):
                self.overwritable = self_outer.overwritable
                self.cached = self_outer.cached

        self_outer = self
        return Descriptor()
    
    
class InputMeta(type):
    def __new__(mcs, name, bases, class_dict):
        cls = super().__new__(mcs, name, bases, class_dict)

        # Collect inherited InputProperty descriptors
        cls._input_properties = {}
        for base in reversed(cls.__mro__):
            for k, v in base.__dict__.items():
                if isinstance(v, InputProperty):
                    cls._input_properties[k] = v

        return cls


class InputTracker(metaclass=InputMeta):
    def __init__(self, **kwargs):
        unknown_keys = set(kwargs) - set(self._input_properties)
        if unknown_keys:
            raise ValueError(f"Unknown input(s): {', '.join(unknown_keys)}")

        self.inputs = {}
        for key, prop in self._input_properties.items():
            if prop.required and key not in kwargs:
                raise ValueError(f"Missing required input: '{key}'")
            value = kwargs.get(key, prop.default)
            setattr(self, key, value)
            self.inputs[key] = value