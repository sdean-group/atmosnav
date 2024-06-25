import abc
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class JaxTree(abc.ABC):
    """ Represents *immutable* objects that have a conversion to a PyTree representation. """

    def __init_subclass__(cls):
        super().__init_subclass__()
        register_pytree_node_class(cls)

    @abc.abstractmethod
    def tree_flatten(self): pass

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children): pass
