import gt4py
import numpy as np

BACKEND = "gtc:gt:cpu_ifirst"

# Cast a dictionary of gt4py storages into dictionary of numpy arrays
def view_gt4py_storage(gt4py_dict):

    np_dict = {}

    for var in gt4py_dict:

        data = gt4py_dict[var]

        # ~ if not isinstance(data, np.ndarray): data.synchronize()
        if BACKEND == "gtcuda":
            data.synchronize()

        if var != 'tauaer' and var != 'taucld':
            np_dict[var] = np.squeeze(data.view(np.ndarray))
        else:
            tmp = np.squeeze(data.view(np.ndarray))
            np_dict[var] = np.transpose(tmp, (0, 2, 1))



    return np_dict

def compare_data(data, ref_data, explicit=True, blocking=True):

    wrong = []
    flag = True

    for var in data:

        if not np.allclose(
            data[var], ref_data[var], rtol=1e-11, atol=1.0e-13, equal_nan=True
        ):

            wrong.append(var)
            flag = False

        else:

            if explicit:
                print(f"Successfully validated {var}!")

    if blocking:
        assert flag, f"Output data does not match reference data for field {wrong}!"
    else:
        if not flag:
            print(f"Output data does not match reference data for field {wrong}!")