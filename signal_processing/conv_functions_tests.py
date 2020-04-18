import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import my_functions as dz
from my_functions import show


def test_conv2d_3d(test_params, prints=False, plot_random=False):

    x_min, x_max, (x_min_height, x_min_width), (x_max_height, x_max_width) = test_params["input"].values()
    k_min, k_max, (k_min_height, k_min_width), (k_max_height, k_max_width) = test_params["kernel"].values()
    min_depth = test_params["min_depth"]
    max_depth = test_params["max_depth"]

    k_num = test_params["k_num"]
    k_stride = 1

    i = 0
    n_tests = (x_max_height + 1 - x_min_height) * (x_max_width + 1 - x_min_width) * (k_max_height + 1 - k_min_height) * (k_max_width + 1 - k_min_width) * (max_depth + 1 - min_depth)
    errors = []

    if plot_random:

        random_indices = np.random.choice(n_tests, 4, replace=False)
        dz_outputs = []
        sp_outputs = []
        shapes = []


    for x_height in range(x_min_height, x_max_height + 1):

        for x_width in range(x_min_width, x_max_width + 1):

            for k_height in range(k_min_height, k_max_height + 1):

                for k_width in range(k_min_width, k_max_width + 1):

                    for depth in range(min_depth, max_depth + 1):

                        x_shape = (depth, x_height, x_width)
                        k_shape = (k_num, depth, k_height, k_width)

                        x = np.random.normal(x_min, x_max + 1, x_shape)
                        k = np.random.normal(k_min, k_max + 1, k_shape)

                        h_pad = k.shape[2] - 1
                        w_pad = k.shape[3] - 1

                        dz_y = dz.conv2d_3d(x, k, k_stride = k_stride, pad=(h_pad, w_pad))

                        sp_y = np.zeros((dz_y.shape))

                        for n in range(k.shape[0]):

                            for d in range(k.shape[1]):

                                sp_y[n, :, :] += sp.convolve2d(x[d, :, :], k[n, d, :, :])

                        errors.append(np.max(np.abs(np.asarray(dz_y - sp_y))))

                        if prints:

                            print(f"\nTest {i + 1}/{n_tests}")
                            print(f"Input Shape:  {x.shape}")
                            print(f"Kernel Shape: {k.shape}")
                            print(f"\nMax Error: {errors[i]}")

                        if plot_random:

                            if i in random_indices:

                                dz_outputs.append(dz_y)
                                sp_outputs.append(sp_y)
                                shapes.append([x.shape, k.shape, dz_y.shape])

                        i += 1

    max_error = np.max(np.abs(np.asarray(errors)))

    print(f"\nCompleted Tests: {i}/{n_tests}")
    print(f"Max Error: {max_error}")

    fig, ax = plt.subplots(3)
    ax[0].imshow(x.transpose(1, 2, 0))
    ax[1].imshow(dz_y[0, :, :])
    ax[2].imshow(sp_y[0, :, :])

    fig, ax = plt.subplots(1)

    ax.plot(errors)

    if max_error > 0:

        ax.set_ylim(0, max_error * 2)

    # plt.show()

    if plot_random:

        fig, ax = plt.subplots(2, figsize=(10, 5), ncols=2)

        j = 0

        for r in range(2):

            for c in range(2):

                if j == len(dz_outputs):
                    break

                ax[r, c].plot(dz_outputs[j].flatten())
                ax[r, c].plot(sp_outputs[j].flatten(), ls='--')
                ax[r, c].set_title(f"x.shape: {shapes[j][0]}, k.shape: {shapes[j][1]}, y.shape: {shapes[j][2]}", fontsize=10)
                j += 1


        fig.tight_layout()

        plt.show()
        plt.ion()



    return None


test_params = {"input":   {"min": 0,
                           "max": 1,
                           "min_shape": (1, 3),
                           "max_shape": (10, 10)
                          },
               "kernel":  {"min": -1,
                           "max": 1,
                           "min_shape": (1, 1),
                           "max_shape": (3, 3)
                          },
               "min_depth": 1,
               "max_depth": 3,
               "k_num": 2
               }

# run_test = True
run_test = False

if run_test == True:

    test_conv2d_3d(test_params, prints=False, plot_random=True)


def test_conv1d_1d(max_input_size=None, max_kernel_size=None):

    # Input and Kernel max sizes
    max_input_size = int(np.random.randint(2, 100, 1))
    max_kernel_size = int(np.random.randint(2, 100, 1))

    prints = False

    i = 0
    all_checks = []
    all_errors = []

    for input_size in range(1, max_input_size):

        for kernel_size in range(1, max_kernel_size):

            print(f"\nTEST {i}: {(input_size, kernel_size)}:")
            print("Input Even" if input_size % 2 == 0 else "Input Odd", "Kernel Even" if kernel_size % 2 == 0 else "Kernel Odd")
            print(f"Input Size = {input_size}, Kernel Size = {kernel_size}")

            # Min and Max values for Input and Kernel
            x = np.random.uniform(0, 10, input_size).astype(np.int32)
            k = np.random.uniform(0, 10, kernel_size).astype(np.int32)

            y = dz.conv1d(x, k, prints)
            np_y = np.convolve(x, k)

            error = np.max(np.abs(y - np_y))
            all_errors.append(error)

            # show(np_y, "Numpy Conv Output")
            # j = 4
            # check = np.array_equal(np.round(y, j), np.round(np_y, j))
            # all_checks.append(check)
            # print(check)
            # assert check is True, f"\nTEST {(input_size, kernel_size)} FAILED: dz.conv1d output IS NOT EQUAL to np.convolve output"
            # print(f"TEST {(input_size, kernel_size)} PASSED: dz.conv1d output IS EQUAL to np.convolve output")
            # print("-" * 100)

            i += 1

    print(f"\n{i} Tests Completed")
    print(f"Maximum Input Size = {max_input_size - 1}")
    print(f"Maximum Kernel Size = {max_kernel_size - 1}")
    print(f"Maximum Error = {np.max(all_errors)}")
    # if all(all_checks) == True:
    #     print("All tests PASSED")

    # elif any(all_checks) == False:
    #     print("Some tests FAILED")

    fig, ax = plt.subplots()

    ax.plot(all_errors)
    ax.set_ylim(0, np.max(all_errors)*2)
    ax.vlines(np.arange(0, i, max_kernel_size), 0, np.max(all_errors)*2, ls='--', lw=0.5, color='r')
    ax.axhline(0, c='r', ls='--')
    # ax.set_xlim(-0.01, i)
    plt.show()

    return None



def test_conv2d_2d(test_params):

    # Min, Max x values and shapes
    min_x = test_params["min_x"]
    max_x = test_params["max_x"]
    min_x_shape = test_params["min_x_shape"]
    max_x_shape = test_params["max_x_shape"]

    # Min, Max x values and shapes
    min_k = test_params["min_k"]
    max_k = test_params["max_k"]
    min_k_shape = test_params["min_k_shape"]
    max_k_shape = test_params["max_k_shape"]

    # Test count, errors list
    i = 0
    errors = []

    x_height_range = max_x_shape[0] - min_x_shape[0] + 1
    x_width_range = max_x_shape[1] - min_x_shape[1] + 1

    k_height_range = max_k_shape[0] - min_k_shape[0] + 1
    k_width_range = max_k_shape[1] - min_k_shape[1] + 1

    expected_tests = x_height_range * x_width_range * k_height_range * k_width_range

    # Toggles
    floats = test_params["floats"]
    prints = test_params["prints"]
    plots = test_params["plots"]
    shows = test_params["shows"]
    timings = test_params["timings"]
    broke = False

    if timings == True:

        dz_times = []
        sp_times = []

    # Iteratre through x shapes
    for x_height in range(min_x_shape[0], max_x_shape[0] + 1):

        # Check expected tests number and break if needed
        if (expected_tests > 100) and (shows == True):

            print(f"\n{expected_tests} Tests Expected")
            print("Too many expected tests with toggle 'shows' == True. Reduce tests to <= 100")

            broke = True
            break

        for x_width in range(min_x_shape[1], max_x_shape[1] + 1):

            # Iterate through all k shapes
            for k_height in range(min_k_shape[0], max_k_shape[0] + 1):

                for k_width in range(min_k_shape[1], max_k_shape[1] + 1):

                    # Test with Random Floats
                    if floats:

                        x = np.random.uniform(min_x, max_x, (x_height, x_width)).astype(np.float32)
                        k = np.random.uniform(min_k, max_k, (k_height, k_width)).astype(np.float32)

                    # Test with Random Ints
                    else:

                        x = np.random.randint(min_x, max_x, (x_height, x_width)).astype(np.int32)
                        k = np.random.randint(min_k, max_k, (k_height, k_width)).astype(np.int32)

                    # Convolutions

                    if timings == True:

                        start_time = time.time()
                        dz_y = dz.conv2d(x, k, shows)
                        dz_times.append(time.time() - start_time)


                        start_time = time.time()
                        sp_y = sp.convolve2d(x, k)
                        sp_times.append(time.time() - start_time)

                    else:

                        dz_y = dz.conv2d(x, k, shows)
                        sp_y = sp.convolve2d(x, k)

                    # Error Calculation
                    errors.append(np.max(np.abs(dz_y - sp_y)))

                    # Prints
                    if prints:

                        print(f"\nTest {i}")
                        print(f"MAX ERROR = {np.max(np.asarray(errors)[i])}")

                    # Shows
                    if shows:

                        show(dz_y, "DZ Conv2d Output")
                        show(sp_y, "SPS Conv2d Output")

                    # Test counter
                    i += 1

    # Prints
    print(f"\n{expected_tests} Tests Expected")
    print(f"{i} Tests Completed")

    if broke == False:

        print(f"\nMin X Shape = {min_x_shape}")
        print(f"Max X Shape = {max_x_shape}")
        print(f"Min K Shape = {min_k_shape}")
        print(f"Max K Shape = {max_k_shape}")

        print(f"\nDZ Conv2d Output Dtype = {dz_y.dtype}")
        print(f"SP Conv2d Output Dtype = {sp_y.dtype}")

        print(f"\nMAX ERROR ACROSS ALL TESTS = {np.max(np.asarray(errors))}")

        # Timings
        if timings:
            dz_av_time = np.mean(np.asarray(dz_times))
            sp_av_time = np.mean(np.asarray(sp_times))
            print(f"\nDZ Conv2d Average Time = {(dz_av_time * 1e3):.4f} ms")
            print(f"SP Conv2d Average Time = {(sp_av_time * 1e3):.4f} ms")
            print(f"Average Time Ratio = {(dz_av_time / sp_av_time):.1f}")

        # Plots
        if plots:

            fig, ax = plt.subplots(2)
            ax[0].plot(np.asarray(errors))

            ax[1].plot(np.asarray(dz_times) * 1e3)
            ax[1].plot(np.asarray(sp_times) * 1e3)
            ax[1].legend(["DZ Timings", "SP Timings"])
            ax[1].set_ylabel("Time (ms)")
            ax[1].set_xlabel("Test Indices")
            ax[1].set_title(f"Max DZ Time = {(np.max(np.asarray(dz_times)) * 1e3):.4f} ms, Max SP Time = {(np.max(np.asarray(sp_times)) * 1e3):.4f} ms")
            print(f"Max DZ Time occured at Test # {np.where(dz_times == np.max(np.asarray(dz_times)))[0][0]}")

            if np.max(np.asarray(errors)) > 0:
                ax[0].set_ylim(0, 2 * np.max(np.asarray(errors)))

            ax[0].set_title(f"Max Error Across All Tests = {np.max(np.asarray(errors))}")
            ax[0].set_ylabel("Error")
            ax[0].set_xlabel("Test Indices")

            fig.tight_layout()

            plt.show()

    return None


# test_params = {"min_x": 0,
#                "max_x": 1,
#                "min_x_shape": (1, 1),
#                "max_x_shape": (5, 5),
#                "min_k": 0,
#                "max_k": 3,
#                "min_k_shape": (1, 1),
#                "max_k_shape": (3, 3),
#                "floats": True,
#                "prints": False,
#                "plots": True,
#                "shows": False,
#                "timings": True
#               }
