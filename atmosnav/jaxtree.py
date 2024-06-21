import abc
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class JaxTree(abc.ABC):

    def __init_subclass__(cls):
        super().__init_subclass__()
        register_pytree_node_class(cls)

    @abc.abstractmethod
    def tree_flatten(self):
        """ """

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data, children):
        """ """
