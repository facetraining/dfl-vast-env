import copy
from core.leras import nn
tf = nn.tf

class OptimizerBase(nn.Saveable):
    def __init__(self, name=None):
        super().__init__(name=name)

    # Define a method for gradient cropping
    def tf_clip_norm(self, g, c, n):
        """Clip the gradient `g` if the L2 norm `n` exceeds `c`.
        # Arguments
            g: Tensor, the gradient tensor
            c: float >= 0. Gradients will be clipped
                when their L2 norm exceeds this value.
            n: Tensor, actual norm of `g`.
        # Returns
            Tensor, the gradient clipped if required.
        """
        # If the cropping paradigm c is less than or equal to 0, there is no need to add the operation to the computation graph
        if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
            return g

        # Determine if the gradient paradigm exceeds a threshold
        condition = n >= c
        # Trimming if thresholds are exceeded
        then_expression = tf.scalar_mul(c / n, g)
        # Otherwise, leave it as it is
        else_expression = g

        # saving the shape to avoid converting sparse tensor to dense
        if isinstance(then_expression, tf.Tensor):
            g_shape = copy.copy(then_expression.get_shape())
        elif isinstance(then_expression, tf.IndexedSlices):
            g_shape = copy.copy(then_expression.dense_shape)
        
        # Ensure that the condition is of type Boolean
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, 'bool')
        
        # Select whether to perform cropping or leave it as is based on conditions
        g = tf.cond(condition,
                    lambda: then_expression,
                    lambda: else_expression)
        
        # Set the shape of the cropped tensor
        if isinstance(then_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(then_expression, tf.IndexedSlices):
            g._dense_shape = g_shape

        return g

# Assign OptimizerBase class to OptimizerBase in nn module
nn.OptimizerBase = OptimizerBase
