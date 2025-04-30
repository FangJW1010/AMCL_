import random
import numpy as np
def load_data(file_path):
    """
    Load data from the given file path.
    Each pair of lines: the first line is the label, the second line is the sequence.
    """
    data = []
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            if lines[i].strip():  # Ensure label line is not empty
                label = np.array([int(x) for x in lines[i].strip()[1:]])
                sequence = lines[i + 1].strip() if i + 1 < len(lines) else ''
                if sequence:  # Ensure sequence line is not empty
                    labels.append(label)
                    data.append(sequence)
    return data, labels


def augment_peptide_dataset_with_reversals_and_back_translation(data, labels, N):
    """
    Augment peptide dataset including reversals, random shuffling, and back-translation enhancement.
    Args:
        data: Original peptide sequence list.
        labels: Original label list.
        N: Number of masked and reversed sample pairs generated for each peptide.
    Returns:
        augmented_peptides: List of augmented peptides.
        augmented_labels: List of augmented labels.
    """
    codon_table = {
        'A': ['GCU', 'GCC', 'GCA', 'GCG'],
        'C': ['UGU', 'UGC'],
        'D': ['GAU', 'GAC'],
        'E': ['GAA', 'GAG'],
        'F': ['UUU', 'UUC'],
        'G': ['GGU', 'GGC', 'GGA', 'GGG'],
        'H': ['CAU', 'CAC'],
        'I': ['AUU', 'AUC', 'AUA'],
        'K': ['AAA', 'AAG'],
        'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
        'M': ['AUG'],
        'N': ['AAU', 'AAC'],
        'P': ['CCU', 'CCC', 'CCA', 'CCG'],
        'Q': ['CAA', 'CAG'],
        'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
        'T': ['ACU', 'ACC', 'ACA', 'ACG'],
        'V': ['GUU', 'GUC', 'GUA', 'GUG'],
        'W': ['UGG'],
        'Y': ['UAU', 'UAC'],
    }
    reverse_codon_table = {v: k for k, values in codon_table.items() for v in values}
    # Define amino acid similarity dictionary
    # Grouped by chemical properties of side chains
    similar_amino_acids = {
        'A': ['G', 'V', 'L'],  # Small nonpolar
        'R': ['K', 'H'],  # Basic amino acids
        'N': ['Q'],  # Acidic neutral
        'D': ['E'],  # Acidic
        'C': ['S', 'T'],  # Thiol/hydroxyl
        'Q': ['N'],  # Acidic neutral
        'E': ['D'],  # Acidic
        'G': ['A', 'V', 'L'],  # Small nonpolar
        'H': ['R', 'K'],  # Basic amino acids
        'I': ['L', 'V'],  # Branched chain nonpolar
        'L': ['I', 'V'],  # Branched chain nonpolar
        'K': ['R', 'H'],  # Basic amino acids
        'M': ['I', 'L', 'V'],  # Large nonpolar
        'F': ['Y', 'W'],  # Aromatic
        'P': ['G'],  # Cyclic
        'S': ['C', 'T'],  # Thiol/hydroxyl
        'T': ['S', 'C'],  # Thiol/hydroxyl
        'W': ['F', 'Y'],  # Aromatic
        'Y': ['F', 'W'],  # Aromatic
        'V': ['A', 'G', 'L'],  # Small nonpolar
    }

    # Ensure each amino acid has at least one similar replacement
    for aa in codon_table.keys():
        if aa not in similar_amino_acids or not similar_amino_acids[aa]:
            similar_amino_acids[aa] = [aa]

    augmented_peptides = []
    augmented_labels = []
    i = 0
    for peptide, peptide_label in zip(data, labels):
        # Add original sequence and label
        augmented_peptides.append(peptide)
        augmented_labels.append(peptide_label[:])
        positions_to_check = list(range(21))  # Check the first 21 positions

        # Use any() function to determine if augmentation is needed
        should_augment = any(peptide_label[pos] == 1 for pos in positions_to_check if pos < len(peptide_label))

        if should_augment and np.sum(peptide_label) == 1:
            i += 1
            # Generate reversed sequence
            augmented_peptides.append(peptide[::-1])
            augmented_labels.append(peptide_label[:])

            for _ in range(N - 1):
                # Randomly select a position for replacement
                if len(peptide) > 0:
                    mask_position = random.randint(0, len(peptide) - 1)
                    original_aa = peptide[mask_position]
                    # Get list of similar amino acids, excluding the original
                    possible_replacements = [aa for aa in similar_amino_acids.get(original_aa, []) if aa != original_aa]
                    if possible_replacements:
                        new_aa = random.choice(possible_replacements)
                        masked_peptide = peptide[:mask_position] + new_aa + peptide[mask_position + 1:]
                        augmented_peptides.append(masked_peptide)
                        augmented_labels.append(peptide_label[:])  # Copy the label


            mrna_sequence = []
            for amino_acid in peptide:
                if amino_acid in codon_table:
                    codon = random.choice(codon_table[amino_acid])
                    mrna_sequence.append(codon)
            mrna_sequence = ''.join(mrna_sequence)

            # Ensure mRNA sequence length is at least 3 bases
            if len(mrna_sequence) >= 3:
                # Ensure there are other non-synonymous codons for replacement
                replacement_attempts = 0
                while True:
                    replacement_position = random.randint(0, len(mrna_sequence) // 3 - 1)
                    codon_start = replacement_position * 3
                    original_codon = mrna_sequence[codon_start:codon_start + 3]

                    if original_codon in reverse_codon_table:
                        # Original codon's amino acid
                        original_amino_acid = reverse_codon_table[original_codon]

                        # Find all different amino acids and their corresponding codons
                        different_codons = []
                        for amino_acid, codons in codon_table.items():
                            if amino_acid != original_amino_acid:  # Exclude the original amino acid
                                different_codons.extend(codons)

                        # If there are available different codons
                        if different_codons:
                            new_codon = random.choice(different_codons)
                            mrna_sequence = mrna_sequence[:codon_start] + new_codon + mrna_sequence[codon_start + 3:]
                            break

                    # Prevent infinite loop, try at most 10 times
                    replacement_attempts += 1
                    if replacement_attempts >= 10:
                        break

            # Translate mRNA sequence back to peptide sequence
            translated_peptide = ''
            for i in range(0, len(mrna_sequence), 3):
                codon = mrna_sequence[i:i + 3]
                if codon in reverse_codon_table:
                    translated_peptide += reverse_codon_table[codon]

            # Add back-translated replacement sequence and label
            if translated_peptide:
                augmented_peptides.append(translated_peptide)
                augmented_labels.append(peptide_label[:])

    # After all augmentations, randomly shuffle the entire augmented dataset
    augmented_data = list(zip(augmented_peptides, augmented_labels))
    random.shuffle(augmented_data)

    # Then extract peptides and labels separately
    augmented_peptides, augmented_labels = zip(*augmented_data)

    # Return shuffled peptides and labels
    return list(augmented_peptides), list(augmented_labels)


def save_augmented_dataset(file_path, peptides, labels):
    """
    Save the augmented dataset to the specified file path.
    Args:
        file_path: Path to save the file.
        peptides: List of augmented peptide sequences.
        labels: List of augmented labels.
    """
    with open(file_path, 'w') as file:
        for label, peptide in zip(labels, peptides):
            label_str = ">" + "".join(map(str, label))  # Keep labels in integer one-hot encoding format
            file.write(label_str + "\n")
            file.write(peptide + "\n")

# Set random seed to ensure reproducible results
random.seed(2024)
np.random.seed(2024)
# Main process
input_file = 'TP_test/dataset/train.txt'  # Path to input file
output_file = 'TP_test/dataset/augmented_train.txt'  # Path to save the augmented dataset
# Load original data
data, labels = load_data(input_file)
# Perform data augmentation
N = 3
augmented_peptides, augmented_labels = augment_peptide_dataset_with_reversals_and_back_translation(data, labels, N)
# Save augmented dataset
save_augmented_dataset(output_file, augmented_peptides, augmented_labels)
print(f"Data augmentation completed, augmented dataset saved to '{output_file}'")