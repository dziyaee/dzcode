import pkgutil
import inspect
from inspect import getmembers


def printmods(package):

    print(f"\nAll Packages & Modules in {package.__name__}:")
    for _, modname, ispkg in pkgutil.walk_packages(path=package.__path__, prefix=package.__name__ + '.', onerror=lambda x: None):

        if ispkg == True:
            print(f"\nPackage: {modname}")

        else:
            print(f"Module:  {modname}")

    print("-" * 100)

    return None


def printmembers(obj, predicate=None):


    if predicate is not None:
        predicate = getattr(inspect, predicate)

    [print(outputs[0]) for outputs in getmembers(obj, predicate)]

    # for output in outputs:
        # obj_names.append(output[0])

    # return obj_names
    return None

