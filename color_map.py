from typing import Tuple, Union
import matplotlib.pyplot as plt

Color = Union[
    str,  
    Tuple[float, float, float], 
    Tuple[float, float, float, float]  
]


cm = plt.cm.Dark2


colors1 = {
    'kron': cm(0),
    'magnitude': cm(1),
    'swag': cm(2),
    'diag': cm(3),
    'None': 'black',
}

colors2 = {
    'kron': cm(4),
    'magnitude': cm(5),
    'swag': cm(6),
    'diag': plt.cm.Set1(8),
    'None': 'grey',
}


def get_color(method: str) -> Color:
    if not method in colors1.keys():
        raise NotImplementedError
    return colors1.get(method, 'black')

def get_color2(method: str) -> Color:
    if not method in colors2.keys():
        raise NotImplementedError
    return colors2.get(method, 'black')