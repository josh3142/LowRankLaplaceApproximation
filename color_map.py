from typing import Tuple, Union
import matplotlib.pyplot as plt

Color = Union[
    str,  
    Tuple[float, float, float], 
    Tuple[float, float, float, float]  
]


cm = plt.cm.Dark2


colors = {
    'kron': cm(0),
    'magnitude': cm(1),
    'swag': cm(2),
    'diagonal': cm(3),
    'None': cm(4),
}


def get_color(method: str) -> Color:
    return colors.get(method, 'black')
