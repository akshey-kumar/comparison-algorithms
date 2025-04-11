import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from itertools import chain

algorithm = 'BunDLeNet'
for worm_num in range(1,5):

    data_path = 'data/raw/c_elegans/NoStim_Data.mat'
    data = Database(data_path=data_path, dataset_no=worm_num)
    b_names = data.behaviour_names

    # loading embedding
    file_pattern = f'data/generated/comparable_embeddings/{{}}__BunDLeNet_worm_{worm_num}'
    Y0 = np.loadtxt(file_pattern.format('Y0'))
    B = np.loadtxt(file_pattern.format('B')).astype(int)

    attractor_labels = np.zeros_like(B)


    def extract_bouts(B, b):
        bouts = []
        current_bout = []

        for i, val in enumerate(B):
            if val == b:
                current_bout.append(i)  # Add index to the current bout
            else:
                if current_bout:  # If the current bout is not empty, save it
                    bouts.append(current_bout)
                    current_bout = []  # Reset for the next bout

        if current_bout:  # Append the last bout if the B ends with a bout
            bouts.append(current_bout)

        bout_indices = sorted(bouts, key=len)
        next_b = [B[idx[-1] + 1] if idx[-1] + 1 < len(B) else None for idx in bout_indices]
        prev_b = [B[idx[0] - 1] if idx[0] - 1 >= 0 else None for idx in bout_indices]

        return bout_indices, np.array(next_b), np.array(prev_b)

    # for i, bout in enumerate(bout_indices):
    #     print(i, bout)
    # print(next_b)
    # print(prev_b)

    ## blue attractor:
    # reverse 1
    attractor_labels[B == 3] = 2

    # slowing before reverse-1
    bout_indices, next_b, prev_b = extract_bouts(B, 6)
    chosen_b = np.where(next_b==3)[0]
    chosen_bouts = [bout_indices[i] for i in chosen_b]
    chosen_bouts = list(chain.from_iterable(chosen_bouts))
    print(chosen_bouts)
    attractor_labels[chosen_bouts] = 2

    # ventral turn before slowing
    bout_indices, next_b, prev_b = extract_bouts(B, 7)
    chosen_b = np.where(next_b==6)[0]
    chosen_bouts = [bout_indices[i] for i in chosen_b]
    chosen_bouts = list(chain.from_iterable(chosen_bouts))
    print(chosen_bouts)
    attractor_labels[chosen_bouts] = 2

    # sus rev before ventral turn
    bout_indices, next_b, prev_b = extract_bouts(B, 5)
    chosen_b = np.where(next_b==7)[0]
    chosen_bouts = [bout_indices[i] for i in chosen_b]
    chosen_bouts = list(chain.from_iterable(chosen_bouts))
    print(chosen_bouts)
    attractor_labels[chosen_bouts] = 2

    # sus rev after reverse-1
    bout_indices, next_b, prev_b = extract_bouts(B, 5)
    chosen_b = np.where(prev_b==3)[0]
    chosen_bouts = [bout_indices[i] for i in chosen_b]
    chosen_bouts = list(chain.from_iterable(chosen_bouts))
    print(chosen_bouts)
    attractor_labels[chosen_bouts] = 2

    # red attractor:
    # reverse 2
    attractor_labels[B == 4] = 1

    # slowing before reverse-2
    bout_indices, next_b, prev_b = extract_bouts(B, 6)
    chosen_b = np.where(next_b==4)[0]
    chosen_bouts = [bout_indices[i] for i in chosen_b]
    chosen_bouts = list(chain.from_iterable(chosen_bouts))
    print(chosen_bouts)
    attractor_labels[chosen_bouts] = 1

    # forward
    attractor_labels[B == 1] = 1

    # dorsal turn
    attractor_labels[B == 0] = 1

    # sus rev before dorsal turn
    bout_indices, next_b, prev_b = extract_bouts(B, 5)
    chosen_b = np.where(next_b==0)[0]
    chosen_bouts = [bout_indices[i] for i in chosen_b]
    chosen_bouts = list(chain.from_iterable(chosen_bouts))
    print(chosen_bouts)
    attractor_labels[chosen_bouts] = 1

    # sus rev after rev-2
    bout_indices, next_b, prev_b = extract_bouts(B, 5)
    chosen_b = np.where(prev_b==4)[0]
    chosen_bouts = [bout_indices[i] for i in chosen_b]
    chosen_bouts = list(chain.from_iterable(chosen_bouts))
    print(chosen_bouts)
    attractor_labels[chosen_bouts] = 1


    attractor_labels = attractor_labels.astype(int)
    np.savetxt(f'data/generated/comparable_embeddings/attractor_labels_worm_{worm_num}.csv', attractor_labels)
    attractor_labels = np.loadtxt(f'data/generated/comparable_embeddings/attractor_labels_worm_{worm_num}.csv')
    file_pattern = f'data/generated/comparable_embeddings/{{}}__BunDLeNet_worm_{worm_num}'
    Y0 = np.loadtxt(file_pattern.format('Y0'))
    B = np.loadtxt(file_pattern.format('B')).astype(int)

    print(b_names)

    # Discrete variable plotting
    vis = LatentSpaceVisualiser(
        y=Y0,
        b=B.astype(int),
        b_names=b_names,
    )
    vis.plot_phase_space(arrow_length_ratio=0.5, show_fig=False)

    vis = LatentSpaceVisualiser(
        y=Y0,
        b=attractor_labels.astype(int),
        b_names=['none', 'red', 'blue'],
    )
    vis.plot_phase_space(arrow_length_ratio=0.5)


    plt.show()
