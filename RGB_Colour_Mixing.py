import numpy as np
from rich import print as rprint
import itertools
from typing import Optional, Tuple



#Implementing linear regression to find the least squared value
def best_linear_approx(
    target: np.ndarray,
    components: np.ndarray,
) -> Tuple[np.ndarray, float]:
    coeffs, costs, _, _ = np.linalg.lstsq(components, target, rcond=None)
    return coeffs, np.sum(np.abs(costs))


def decompose(
        
    color,
    colors,
    max_nums: int = 3,
    min_weights: float = 0.0,
    max_weights: float = 1.0,
    acc_to: Optional[float] = None,
    use_avg = False,
    force_acc: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float, Optional[float]]:
    """Decompose `color` into a linear combination of `n` colors.

    The decomposition is subject to some constraints.

    Args:
        color: The color to decompose.
        colors: The base colors to use for the decomposition.
        max_nums: The maximum number of base colors to use.
        min_weights: The minimum value for the weights.
        max_weights: The maximum value for the weights.
        acc_to: The value the weights should accumulate (sum or average) to.
        use_avg: Use average instead of sum for composition / decomposition.
        force_acc: If True, the weights are normalized to `acc_to`, if set.

    Return:
        best_weights, approx_color, best_indices, best_selected, best_cost, acc_weights


    """
    color = np.array(color)
    colors = np.array(colors).T
    num_channels, num_colors = colors.shape

    # This is only used if u want the sum of weights to be >=1 or 2
    if acc_to is not None:
        colors = np.concatenate(
            [colors, np.ones(num_colors, dtype=colors.dtype)[None, ...]], axis=0
        )
        color = np.concatenate([color, np.full(1, acc_to, dtype=colors.dtype)])
    # Loop to find the best index with best weights     
    for n in range(1, max_nums + 1):
        best_indices = None
        best_weights = None
        best_cost = np.inf
        # loop iterates through all the colors to find the best combinations indices that can be used
        for indices in itertools.combinations(range(num_colors), n):
            if use_avg:
                color *= n
            weights, curr_cost = best_linear_approx(color, colors[:, indices])
            # if u give condition for weights this loop is used to find best combinations
            if force_acc and acc_to is not None:
                weights *= acc_to / np.sum(weights)
            if use_avg:
                weights *= n
            if (
                curr_cost < best_cost
                and (min_weights is None or np.all(weights >= min_weights))
                and (max_weights is None or np.all(weights <= max_weights))
            ):
                best_indices = indices
                best_weights = weights
                best_cost = curr_cost
            if use_avg:
                color /= n
    # used to select the colors present in those indices            
    if best_indices is not None:
        approx_color = (colors[:, best_indices] @ best_weights)[:num_channels]
        acc_weights = np.sum(best_weights)
        if use_avg:
            n = len(best_indices)
            approx_color /= n
            acc_weights /= n
        best_selected = colors[:num_channels, best_indices].T
    # when the code can't find a proper match this executes    
    else:
        approx_color = None
        best_selected = None
        acc_weights = None
    return best_weights, approx_color, best_indices, best_selected, best_cost, acc_weights


def main():
    """main function to return the the 1 to 3 colours that can be used to form the required color

    Returns:
        colors: the three colours RGB code and the percentage
    """
    #input color
    color = []
    for i in range(0,3):
        ip = float(input())
        color.append(ip)
    #colors in the system
    colors = np.load('base_colors.npz')['arr_0']

    rprint(decompose(color,colors))    
    
    return decompose(color,colors)


if __name__ == '__main__':
    main()