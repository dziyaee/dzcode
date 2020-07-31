# # Alternate implementation of Shape Class using Dimension Class as Shape Class attributes
# class Dimension:
#     def __init__(self, n):
#         self.n = n

#     def __get__(self, obj, cls):
#         if obj is None:
#             return self
#         try:
#             return obj.shape[self.n]
#         except IndexError:
#             return None

#     def __set__(self, obj, value):
#         raise AttributeError("can't set attribute")


# class Shape:
#     width = Dimension(-1)
#     height = Dimension(-2)
#     depth = Dimension(-3)
#     num = Dimension(-4)

#     def __init__(self, shape):
#         self.shape = shape

#     @property
#     def shape(self):
#         return self._shape

#     @shape.setter
#     def shape(self, shape_):
#         if not isinstance(shape_, (tuple, list)):
#             raise TypeError(f"Expected tuple or list, got {type(shape_)}")

#         if len(shape_) == 0:
#             raise ValueError(f"Expected a tuple or list of length > 0, got {len(shape_)}")

#         if  not all(isinstance(dim, int) for dim in shape_):
#             raise TypeError(f"Expected only integer elements, got {shape_}")

#         if not all(dim > 0 for dim in shape_):
#             raise ValueError(f"Expected only positive integer elements, got {shape_}")

#         self._shape = tuple(shape_)

#     @property
#     def ndim(self):
#         return len(self.shape)
