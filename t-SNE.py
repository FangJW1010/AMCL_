import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn import manifold
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from adjustText import adjust_text

# Configuration
max_len = 50
min_len = 5
batch_size = 64 * 4
vocab_size = 21
embedding_size = 128
filter_num = 128
filter_size = [3, 4, 5]
output_size = 21
dropout = 0.6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Predefine all constants
categories = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP',
              'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']

unique_func_labels = ['THP', 'ADP', 'DPPIP', 'BBP', 'AHP', 'AIP', 'ADP_AIP', 'ACP', 'BIP', 'CPP', 'ADP_DPPIP',
                      'SBP', 'AAP', 'ABP_ACP_AFP', 'ATP', 'ABP', 'ABP_AFP_APP', 'AHP_DPPIP', 'APP', 'AFP',
                      'QSP', 'ABP_AFP', 'ABP_ACP', 'ACVP', 'AVP']

cmap = ['#314a9d', '#747cb5', '#b2b5cd', '#7f1e40', '#ae6480', '#ccabb7',
        '#4cb64b', '#83cb85', '#b9d9b8', '#e3832b', '#f2c8ad', '#e2c4af',
        '#c5306b', '#d4618c', '#dd96af', '#54c0ac', '#000000', '#ff0000',
        '#006400', '#ffff00', '#93d6c6', '#6bcada', '#efd3e4', '#a5b472', '#75787f']


def find_density_center(points, bandwidth=0.5):
    """
    Find the point with highest density in a cluster using KDE with optimized bandwidth
    """
    # If too few points, return the mean directly
    if len(points) < 5:
        return np.mean(points, axis=0)

    # Use smaller bandwidth for more precise density center
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(points)

    # Evaluate density on a denser grid
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    # Create a more detailed grid
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()]).T

    # Calculate density
    density = kde.score_samples(positions)

    # Return the point with highest density
    return positions[np.argmax(density)]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, feature_dim=128):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs, padding='same')
            for fs in filter_sizes
        ])

        hidden_dim = len(filter_sizes) * n_filters

        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 10, hidden_dim * 5),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, data, return_features=True):
        embedded = self.embedding(data)
        embedded = embedded.permute(0, 2, 1)
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        cat = self.dropout(torch.cat(flatten, dim=1))

        features = self.feature_extractor(cat)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


def SeqsData2EqlTensor(file_path, min_len=5, max_len=50):
    aa_dict = {
        '[PAD]': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        'U': 0, 'X': 0, 'J': 0
    }

    with open(file_path, 'r') as inf:
        lines = inf.read().splitlines()

    pep_codes = []
    labels = []
    valid_count = 0
    invalid_count = 0
    current_label = None

    for line in lines:
        if line.startswith('>'):
            current_label = [int(i) for i in line[1:]]
        else:
            seq_len = len(line)
            if min_len <= seq_len <= max_len:
                current_pep = []
                for aa in line:
                    if aa.upper() in aa_dict:
                        current_pep.append(aa_dict[aa.upper()])
                if current_pep:
                    pep_codes.append(torch.tensor(current_pep))
                    labels.append(current_label)
                    valid_count += 1
            else:
                invalid_count += 1

    print(f"Total sequences processed: {valid_count + invalid_count}")
    print(f"Valid sequences (length {min_len}-{max_len}): {valid_count}")
    print(f"Invalid sequences: {invalid_count}")

    if not pep_codes:
        raise ValueError("No valid sequences found after filtering!")

    return torch.nn.utils.rnn.pad_sequence(pep_codes, batch_first=True, padding_value=aa_dict['[PAD]']), torch.tensor(
        labels)


def get_label_type(label_vector):
    active_indices = np.where(label_vector == 1)[0]
    if len(active_indices) == 0:
        return "Unknown"

    active_categories = [categories[i] for i in active_indices]
    return '_'.join(sorted(active_categories))


# Main program
if __name__ == "__main__":
    # Load and process data
    print("Loading and processing data...")
    test_data, test_label = SeqsData2EqlTensor('dataset/test.txt', min_len=5, max_len=50)
    test_dataset = Data.TensorDataset(test_data, test_label)
    test_iter = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize and load model
    print("Loading model...")
    model = TextCNN(vocab_size, embedding_size, filter_num, filter_size, output_size, dropout)
    model.load_state_dict(torch.load('saved_models/model_select+TextCNN0.pth', weights_only=True))
    model = model.to(device)
    model.eval()

    # Collect features
    print("Extracting features...")
    features_list = []
    labels_list = []
    with torch.no_grad():
        for seq, lab in test_iter:
            seq = seq.to(device)
            logits, features = model(seq)
            features = F.normalize(features, p=2, dim=1)
            features_list.extend(features.cpu().numpy())
            labels_list.extend(lab.numpy())

    features = np.array(features_list)
    labels = np.array(labels_list)

    # Data standardization
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    # PCA dimensionality reduction
    print("Performing PCA...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)

    # t-SNE dimensionality reduction
    print("Performing t-SNE...")
    tsne = manifold.TSNE(
        n_components=2,
        perplexity=40,
        early_exaggeration=12.0,
        learning_rate=200,
        n_iter=3000,
        min_grad_norm=1e-7,
        metric='euclidean',
        init='pca',
        verbose=1,
        random_state=2023
    )
    features_tsne = tsne.fit_transform(features_pca)

    # Process labels
    label_types = [get_label_type(label) for label in labels]

    # Create visualization
    print("Creating visualization...")
    plt.figure(figsize=(24, 18), dpi=300)  # Increase figure size and DPI

    # Draw scatter plot and prepare annotations
    print("Drawing scatter plot...")
    label_centers = {}
    annotation_positions = {}

    # First draw all scatter points
    for i, label_type in enumerate(unique_func_labels):
        mask = np.array(label_types) == label_type
        if np.any(mask):
            points = features_tsne[mask]
            plt.scatter(points[:, 0], points[:, 1],
                        c=cmap[i],
                        label=label_type,
                        alpha=0.7,
                        s=80,  # Increase scatter point size
                        edgecolors='white',
                        linewidth=0.8)  # Increase border width  0.8

            # Calculate density center
            density_center = find_density_center(points)
            label_centers[label_type] = density_center

            # Optimize text annotation position calculation
            cov = np.cov(points.T)
            eigenvals, eigenvects = np.linalg.eigh(cov)
            main_direction = eigenvects[:, np.argmax(eigenvals)]
            spread = np.sqrt(np.max(eigenvals))
            text_offset = spread * 0.35  # Slightly increase text offset distance

            center_to_mean = np.mean(points, axis=0) - density_center
            if np.dot(main_direction, center_to_mean) < 0:
                main_direction = -main_direction

            text_pos = density_center + main_direction * text_offset
            annotation_positions[label_type] = text_pos

    # Add indicator lines and text annotations
    print("Adding annotation lines...")
    texts = []
    for label_type in label_centers:
        center = label_centers[label_type]
        text_pos = annotation_positions[label_type]

        # Optimize indicator line style
        mid_point = (center + text_pos) / 2
        control_point = mid_point + np.array([-1, 1]) * 2

        t = np.linspace(0, 1, 100)
        curve_x = (1 - t) ** 2 * center[0] + 2 * (1 - t) * t * control_point[0] + t ** 2 * text_pos[0]
        curve_y = (1 - t) ** 2 * center[1] + 2 * (1 - t) * t * control_point[1] + t ** 2 * text_pos[1]

        plt.plot(curve_x, curve_y,
                 '-',
                 color='gray',
                 alpha=0.7,
                 linewidth=1.0,  # Increase indicator line width
                 zorder=1)

        # Add text annotations, increase font size and improve visibility
        text = plt.text(
            text_pos[0], text_pos[1],
            label_type,
            fontsize=25,  # Further increase font size 18
            fontweight='bold',  # Add bold
            alpha=0.95,
            bbox=dict(facecolor='white',
                      edgecolor='lightgray',  # Add border
                      alpha=0.9,
                      pad=2.0,
                      boxstyle='round,pad=0.5'),  # Rounded border
            horizontalalignment='center',
            verticalalignment='center'
        )
        texts.append(text)

    # Optimize text layout parameters
    adjust_text(texts,
                expand_points=(2.2, 2.2),
                force_points=(0.9, 0.9),
                force_text=(0.8, 0.8),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.7, shrinkA=5),
                only_move={'points': 'y', 'text': 'xy'},
                save_steps=False,
                max_iter=100)

    # Set figure style
    plt.grid(True, linestyle='--', alpha=0.3)

    # Show all borders and set style
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.xlabel('Dimension 1', fontsize=30, fontweight='bold')
    plt.ylabel('Dimension 2', fontsize=30, fontweight='bold')
    plt.title('t-SNE Visualization of Peptide Features', fontsize=35, fontweight='bold', pad=20)

    legend_elements = []
    for i, label_type in enumerate(unique_func_labels):
        legend_elements.append(plt.scatter([], [],
                                           c=cmap[i],
                                           s=100,  # Increase point size in legend
                                           label=label_type,
                                           alpha=0.7,
                                           edgecolors='white',
                                           linewidth=0.8))

    # Optimize legend
    # Create legend and precisely control position and spacing
    leg = plt.legend(handles=legend_elements,
                     bbox_to_anchor=(1.0, 0.98),  # Fine-tune top position
                     loc='upper left',
                     ncol=1,
                     fontsize=18,
                     frameon=True,
                     edgecolor='black',
                     fancybox=True,
                     shadow=True,
                     # title='Peptide Types',
                     title_fontsize=20,
                     borderpad=0.5,
                     labelspacing=0.3,
                     handlelength=1.0,
                     handletextpad=0.5,
                     borderaxespad=0)

    # Set legend border to dashed
    leg.get_frame().set_linestyle('--')  # Set to dashed line
    leg.get_frame().set_linewidth(1)  # Set line width
    leg.get_frame().set_edgecolor('gray')  # Set border color to gray

    # Adjust layout to ensure legend is fully aligned with left border
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    # Adjust layout and save
    plt.tight_layout()


    # Adjust layout to ensure legend is fully aligned with left border
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    # Adjust layout and save
    plt.tight_layout()

    # Save high resolution version
    print("Saving visualizations...")

    plt.savefig('tsne_visualization2.svg', format='svg', bbox_inches='tight', facecolor='white')


    print("Visualizations saved successfully!")
    plt.close()