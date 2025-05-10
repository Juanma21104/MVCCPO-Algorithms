import algorithms.utils as utils
import numpy as np

if __name__ == "__main__":
    
    y = np.array([0.2, -0.3, 0.5, 0.8])
    z = utils.projection_probability_simplex(y)
    print("Vector proyectado:", z)
    print("Suma:", np.sum(z))
    print("¿Todos >= 0?:", np.all(z >= 0))
    
