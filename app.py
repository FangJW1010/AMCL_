import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VOCAB_SIZE = 21
EMBEDDING_SIZE = 128
FILTER_NUM = 128
FILTER_SIZE = [3, 4, 5]
OUTPUT_SIZE = 21
DROPOUT = 0.6
MAX_LEN = 50
BATCH_SIZE = 32  # Batch size

# Amino acid dictionary
AA_DICT = {
    '[PAD]': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'U': 0, 'X': 0, 'J': 0
}

# Valid amino acids set (for validation)
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

CATEGORIES = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP',
              'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']

# Category map
CATEGORY_MAP = {
    'AAP': 'Anti-angiogenic Peptide',
    'ABP': 'Anti-bacterial Peptide',
    'ACP': 'Anti-cancer Peptide',
    'ACVP': 'Anti-coronavirus Peptide',
    'ADP': 'Anti-diabetic Peptide',
    'AEP': 'Anti-endotoxin Peptide',
    'AFP': 'Anti-fungal Peptide',
    'AHIVP': 'Anti-HIV Peptide',
    'AHP': 'Anti-hypertensive Peptide',
    'AIP': 'Anti-inflammatory Peptide',
    'AMRSAP': 'Anti-MRSA Peptide',
    'APP': 'Anti-parasitic Peptide',
    'ATP': 'Anti-tubercular Peptide',
    'AVP': 'Anti-viral Peptide',
    'BBP': 'Blood-brain Barrier Peptide',
    'BIP': 'Biofilm-inhibitory Peptide',
    'CPP': 'Cell-penetrating Peptide',
    'DPPIP': 'Dipeptidyl Peptidase IV Peptide',
    'QSP': 'Quorum-sensing Peptide',
    'SBP': 'Surface Binding Peptide',
    'THP': 'Tumor Homing Peptide'
}

# Category descriptions (for hints)
CATEGORY_DESCRIPTIONS = {
    'AAP': 'Peptides that inhibit angiogenesis by targeting VEGF signaling pathway to prevent tumor vascularization',
    'ABP': 'Peptides that specifically recognize and disrupt bacterial cell membranes, effective against both Gram-positive and Gram-negative bacteria',
    'ACP': 'Peptides that inhibit tumor cell growth and metastasis through multiple mechanisms including apoptosis induction and angiogenesis inhibition',
    'ACVP': 'Peptides that specifically target coronavirus spike protein or proteases to prevent viral entry into host cells',
    'ADP': 'Peptides that regulate blood glucose levels by modulating insulin secretion or improving insulin sensitivity',
    'AEP': 'Peptides that neutralize bacterial endotoxin (LPS) activity to reduce endotoxin-induced inflammatory responses',
    'AFP': 'Peptides that exert antifungal effects by disrupting fungal cell membranes or inhibiting fungal growth',
    'AHIVP': 'Peptides that specifically target HIV viral proteins or receptors to prevent viral replication and infection',
    'AHP': 'Peptides that lower blood pressure by inhibiting angiotensin-converting enzyme (ACE) or modulating vascular tone',
    'AIP': 'Peptides that reduce inflammation by inhibiting pro-inflammatory cytokines or modulating immune responses',
    'AMRSAP': 'Peptides with specific activity against methicillin-resistant Staphylococcus aureus (MRSA)',
    'APP': 'Peptides that disrupt parasite cell membranes or inhibit parasite growth',
    'ATP': 'Peptides that specifically target Mycobacterium tuberculosis to inhibit bacterial growth and replication',
    'AVP': 'Peptides that inhibit viral replication and infection through multiple mechanisms',
    'BBP': 'Peptides capable of crossing the blood-brain barrier for drug delivery to the central nervous system',
    'BIP': 'Peptides that disrupt or inhibit bacterial biofilm formation to enhance antibiotic efficacy',
    'CPP': 'Peptides capable of penetrating cell membranes for intracellular delivery of drugs or biomolecules',
    'DPPIP': 'Peptides that inhibit dipeptidyl peptidase IV (DPP-4) activity for type 2 diabetes treatment',
    'QSP': 'Peptides that interfere with bacterial quorum sensing to inhibit virulence factor expression',
    'SBP': 'Peptides that specifically bind to particular surfaces or molecules for targeted delivery or detection',
    'THP': 'Peptides that specifically recognize and bind to tumor cells for targeted cancer therapy'
}

# Category colors for visualization
CATEGORY_COLORS = {
    'AAP': '#4299E1',
    'ABP': '#805AD5',
    'ACP': '#48BB78',
    'ACVP': '#F56565',
    'ADP': '#ED8936',
    'AEP': '#ECC94B',
    'AFP': '#38B2AC',
    'AHIVP': '#F687B3',
    'AHP': '#9F7AEA',
    'AIP': '#FC8181',
    'AMRSAP': '#4FD1C5',
    'APP': '#667EEA',
    'ATP': '#B794F4',
    'AVP': '#7F9CF5',
    'BBP': '#F6AD55',
    'BIP': '#68D391',
    'CPP': '#ED64A6',
    'DPPIP': '#D6BCFA',
    'QSP': '#90CDF4',
    'SBP': '#FEB2B2',
    'THP': '#9AE6B4'
}

# Example sequences for user reference
EXAMPLE_SEQUENCES = [
    "GLFDIVKKVVGALG",  # Common antimicrobial peptide sequence
    "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin 2
    "RWCFRVCYRGICYRRCR",  # Anti-viral peptide example
    "GRKKRRQRRRPPQ",  # HIV-TAT, cell-penetrating peptide
    "DRVIEVLQKAGAQ"  # Anti-hypertensive peptide example
]


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: list, output_dim: int,
                 dropout: float, feature_dim: int = 128):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])

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

    def forward(self, data, return_features=False, length=None):
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


# Helper functions
def validate_sequence(sequence):
    """Validate if the amino acid sequence is valid"""
    if not sequence or not isinstance(sequence, str):
        return False
    return all(aa in VALID_AA for aa in sequence.upper())


def process_sequence(sequence):
    """Process the sequence into a model input format"""
    # Convert sequence to numbers
    indices = [AA_DICT.get(aa, 0) for aa in sequence]
    # Pad to fixed length
    if len(indices) < MAX_LEN:
        indices += [0] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]
    return torch.tensor(indices).unsqueeze(0)


def process_batch(sequences, batch_size=BATCH_SIZE):
    """Process sequences in batches"""
    all_results = {}
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        batch_tensors = []

        for seq in batch_sequences:
            # Convert sequence to numbers
            indices = [AA_DICT.get(aa, 0) for aa in seq]
            # Pad to fixed length
            if len(indices) < MAX_LEN:
                indices += [0] * (MAX_LEN - len(indices))
            else:
                indices = indices[:MAX_LEN]
            batch_tensors.append(indices)

        # Convert to batch tensor
        batch_tensor = torch.tensor(batch_tensors).to(device)

        # Prediction
        with torch.no_grad():
            logits = model(batch_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        # Collect results for each sequence
        for j, seq in enumerate(batch_sequences):
            results = {
                category: float(prob)
                for category, prob in zip(CATEGORIES, probabilities[j])
            }
            all_results[seq] = results

        logger.info(f"Processed batch {i // batch_size + 1}/{total_batches} ({len(batch_sequences)} sequences)")

    return all_results


def predict_single_sequence(sequence):
    """Predict function for a single sequence"""
    sequence = sequence.strip().upper()

    # Validate sequence
    if not validate_sequence(sequence):
        return None, "Invalid amino acid sequence. Only standard amino acids (ACDEFGHIKLMNPQRSTVWY) are allowed."

    # Process input sequence
    input_tensor = process_sequence(sequence).to(device)

    # Prediction
    start_time = time.time()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Pack results into a dictionary
    results = {
        category: float(prob)
        for category, prob in zip(CATEGORIES, probabilities)
    }

    # Log prediction time
    elapsed_time = time.time() - start_time
    logger.info(f"Prediction for sequence {sequence} completed in {elapsed_time:.4f} seconds")

    return results, ""


def process_file_upload(file_obj):
    """Process uploaded file and extract sequences"""
    if file_obj is None:
        return [], "No file uploaded"

    try:
        file_path = file_obj.name
        sequences = []

        # Process based on file type
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if len(df.columns) > 0:
                # Use the first column
                first_col = df.iloc[:, 0]
                sequences = [str(seq).strip().upper() for seq in first_col if str(seq).strip()]
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
            if len(df.columns) > 0:
                first_col = df.iloc[:, 0]
                sequences = [str(seq).strip().upper() for seq in first_col if str(seq).strip()]
        else:
            # Assume it's a text file
            with open(file_path, 'r', encoding='utf-8') as f:
                sequences = [line.strip().upper() for line in f if line.strip()]

        # Validate sequences
        valid_sequences = [seq for seq in sequences if validate_sequence(seq)]

        if not valid_sequences:
            return [], "No valid sequences found in the file"

        invalid_count = len(sequences) - len(valid_sequences)
        message = f"Read {len(valid_sequences)} valid sequences"
        if invalid_count > 0:
            message += f", skipped {invalid_count} invalid sequences"

        return valid_sequences, message

    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return [], f"Error processing file: {str(e)}"


def plot_prediction_results(results, sequence):
    """Create bar chart for prediction results"""
    if not results:
        return None

    # Sort categories by probability
    sorted_data = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_categories = sorted_data[:10]  # Get top 10

    # Extract categories and probabilities
    categories = [CATEGORY_MAP.get(cat, cat) for cat, _ in top_categories]
    probabilities = [prob * 100 for _, prob in top_categories]
    colors = [CATEGORY_COLORS.get(cat, '#718096') for cat, _ in top_categories]

    # Create chart
    plt.figure(figsize=(12, 6))
    bars = plt.barh(categories, probabilities, color=colors)

    # Add percentage labels
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{prob:.1f}%', va='center')

    plt.title(f'Sequence Prediction Results: {sequence[:30]}{"..." if len(sequence) > 30 else ""}')
    plt.xlabel('Probability (%)')
    plt.xlim(0, 105)  # Give space for percentage labels
    plt.tight_layout()

    return plt


def format_detailed_results(results):
    """Format detailed prediction results into HTML"""
    if not results:
        return ""

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 16px;'>"

    for category, probability in sorted_results:
        full_name = CATEGORY_MAP.get(category, category)
        color = CATEGORY_COLORS.get(category, '#718096')
        percentage = probability * 100

        # Add more interactive visual effects
        html += f"""
        <div style='border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s, box-shadow 0.2s;' onmouseover='this.style.transform="translateY(-3px)"; this.style.boxShadow="0 4px 8px rgba(0,0,0,0.15)"' onmouseout='this.style.transform="translateY(0)"; this.style.boxShadow="0 2px 4px rgba(0,0,0,0.1)"'>
            <div style='font-weight: 600; font-size: 1.05em; margin-bottom: 4px;'>{full_name}</div>
            <div style='color: #718096; font-size: 0.85em; margin-bottom: 8px;'>({category})</div>
            <div style='font-size: 1.5em; font-weight: bold; color: {color}; margin-bottom: 8px;'>{percentage:.1f}%</div>
            <div style='background-color: #f1f5f9; height: 10px; border-radius: 5px; overflow: hidden;'>
                <div style='width: {percentage}%; height: 100%; border-radius: 5px; background-color: {color}; transition: width 0.5s ease-out;'></div>
            </div>
            <div style='font-size: 0.8em; color: #4a5568; margin-top: 8px;'>{CATEGORY_DESCRIPTIONS.get(category, "")}</div>
        </div>
        """

    html += "</div>"
    return html


def create_category_table():
    """Create formatted table for category descriptions"""
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); gap: 16px; margin-top: 20px;'>"

    for code, name in CATEGORY_MAP.items():
        color = CATEGORY_COLORS.get(code, '#718096')
        description = CATEGORY_DESCRIPTIONS.get(code, "")

        html += f"""
        <div style='display: flex; align-items: flex-start; padding: 12px; border: 1px solid #e2e8f0; border-radius: 8px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='width: 16px; height: 16px; border-radius: 50%; background-color: {color}; margin-right: 12px; margin-top: 4px; flex-shrink: 0;'></div>
            <div style='flex-grow: 1;'>
                <div style='font-weight: 600; font-size: 1.05em;'>{name}</div>
                <div style='color: #4a5568; font-size: 0.9em; margin-bottom: 4px;'>({code})</div>
                <div style='font-size: 0.9em; color: #4a5568;'>{description}</div>
            </div>
        </div>
        """

    html += "</div>"
    return html


def color_sequence(sequence):
    """Add color highlighting to amino acid sequence"""
    if not sequence:
        return ""

    # Amino acid color codes (grouped by properties)
    aa_colors = {
        # Non-polar (hydrophobic)
        'A': '#FF9E80', 'V': '#FF9E80', 'L': '#FF9E80', 'I': '#FF9E80',
        'M': '#FF9E80', 'F': '#FF9E80', 'W': '#FF9E80', 'P': '#FF9E80',
        # Polar (neutral)
        # Polar (neutral)
        'G': '#80D8FF', 'S': '#80D8FF', 'T': '#80D8FF', 'C': '#80D8FF',
        'Y': '#80D8FF', 'N': '#80D8FF', 'Q': '#80D8FF',
        # Acidic (negative charge)
        'D': '#B388FF', 'E': '#B388FF',
        # Basic (positive charge)
        'K': '#EA80FC', 'R': '#EA80FC', 'H': '#EA80FC'
    }

    # Create color-coded legend
    legend = """
        <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; font-size: 0.85em;'>
            <div style='display: flex; align-items: center;'>
                <div style='width: 12px; height: 12px; background-color: #FF9E80; margin-right: 4px; border-radius: 2px;'></div>
                <span>Non-polar</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <div style='width: 12px; height: 12px; background-color: #80D8FF; margin-right: 4px; border-radius: 2px;'></div>
                <span>Polar</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <div style='width: 12px; height: 12px; background-color: #B388FF; margin-right: 4px; border-radius: 2px;'></div>
                <span>Acidic</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <div style='width: 12px; height: 12px; background-color: #EA80FC; margin-right: 4px; border-radius: 2px;'></div>
                <span>Basic</span>
            </div>
        </div>
        """

    # Add sequence length and composition information
    composition = {}
    for aa in sequence:
        if aa in VALID_AA:
            composition[aa] = composition.get(aa, 0) + 1

    comp_html = "<div style='margin-bottom: 10px; font-size: 0.9em;'>"
    comp_html += f"<span style='font-weight: 600;'>Sequence Length:</span> {len(sequence)} | "
    comp_html += "<span style='font-weight: 600;'>Amino Acid Composition:</span> "
    comp_items = [f"{aa} ({count}, {count / len(sequence) * 100:.1f}%)" for aa, count in sorted(composition.items())]
    comp_html += ", ".join(comp_items)
    comp_html += "</div>"

    html = legend + comp_html

    html += "<div style='font-family: monospace; font-size: 1.2em; letter-spacing: 2px; line-height: 1.6; padding: 12px; background-color: #f8fafc; border-radius: 8px; overflow-x: auto;'>"

    # Display sequence in chunks for readability
    for i in range(0, len(sequence), 10):
        chunk = sequence[i:i + 10]
        html += "<span style='margin-right: 10px;'>"
        for j, aa in enumerate(chunk):
            color = aa_colors.get(aa, '#4a5568')
            html += f"<span style='color: {color}; font-weight: 600;' title='{aa}'>{aa}</span>"
        html += "</span>"

        # Add position markers
        if i + 10 < len(sequence):
            html += f"<span style='color: #a0aec0; font-size: 0.8em; margin-right: 8px;'>{i + 10}</span>"

        if i + 10 < len(sequence):
            html += "<br>"

    html += "</div>"

    return html


# Add batch prediction progress update
def process_batch_prediction(sequences, progress=gr.Progress()):
    """Batch prediction with progress updates"""
    if not sequences:
        return "No valid sequences to process", None, None

    results = {}
    total = len(sequences)

    # Batch processing for efficiency
    batch_size = min(32, total)  # Adjust based on system capabilities
    total_batches = (total + batch_size - 1) // batch_size

    progress(0, desc="Starting batch prediction...")

    for i in range(0, total, batch_size):
        progress((i / total), desc=f"Processing batch {i // batch_size + 1}/{total_batches}...")
        batch = sequences[i:i + batch_size]

        batch_results = process_batch(batch)
        results.update(batch_results)

    progress(1, desc="Batch prediction complete")

    # Create a DataFrame for displaying results
    results_df = []
    for seq, pred in results.items():
        # Find the top prediction
        top_category = max(pred.items(), key=lambda x: x[1])
        top_2nd_category = sorted(pred.items(), key=lambda x: x[1], reverse=True)[1]

        results_df.append({
            "Sequence": seq,
            "Sequence Length": len(seq),
            "Primary Function Prediction": f"{CATEGORY_MAP.get(top_category[0], top_category[0])} ({top_category[0]})",
            "Prob": f"{top_category[1] * 100:.1f}%",
            "Secondary Function Prediction": f"{CATEGORY_MAP.get(top_2nd_category[0], top_2nd_category[0])} ({top_2nd_category[0]})",
            "Secondary Prob": f"{top_2nd_category[1] * 100:.1f}%"
        })

    df = pd.DataFrame(results_df)

    # Create a downloadable CSV of all results
    all_results_df = []
    for seq, pred in results.items():
        row = {"Sequence": seq, "Sequence Length": len(seq)}
        for cat, prob in pred.items():
            row[f"{cat} ({CATEGORY_MAP.get(cat, cat)})"] = prob
        all_results_df.append(row)

    full_df = pd.DataFrame(all_results_df)

    # Return summary message, results table, and full CSV
    return f"Completed predictions for {len(results)} sequences", df, full_df


# Gradio interface function
def predict_tab(sequence, progress=gr.Progress()):
    """Handle single sequence prediction"""
    if not sequence:
        return None, "Please enter a sequence", "", ""

    sequence = sequence.strip().upper()
    progress(0.3, desc="Validating sequence...")

    # Validate sequence
    if not validate_sequence(sequence):
        return None, "Invalid amino acid sequence. Only standard amino acids (ACDEFGHIKLMNPQRSTVWY) are allowed.", "", ""

    progress(0.6, desc="Running prediction...")

    # Prediction
    results, error = predict_single_sequence(sequence)
    if error:
        return None, error, "", ""

    progress(0.9, desc="Generating visualization...")

    # Generate plot and formatted results
    plot = plot_prediction_results(results, sequence)
    detailed_results = format_detailed_results(results)
    colored_seq = color_sequence(sequence)

    # Find Top category
    top_category = max(results.items(), key=lambda x: x[1])[0]
    top_prob = results[top_category] * 100

    success_message = f"Prediction successful! Primary function: {CATEGORY_MAP.get(top_category)} ({top_category}), Probability: {top_prob:.1f}%"

    return plot, success_message, colored_seq, detailed_results


def batch_tab_file(file_obj, progress=gr.Progress()):
    """Handle batch prediction from a file"""
    if file_obj is None:
        return "Please upload a file", None, None

    progress(0.2, desc="Processing file...")
    sequences, message = process_file_upload(file_obj)

    if not sequences:
        return message, None, None

    # Process batch prediction
    return process_batch_prediction(sequences, progress)


def batch_tab_text(text_input, progress=gr.Progress()):
    """Handle batch prediction from text input"""
    if not text_input or text_input.strip() == "":
        return "Please enter at least one sequence", None, None

    progress(0.1, desc="Processing input...")

    # Split input by newline and clean
    lines = [line.strip().upper() for line in text_input.split("\n") if line.strip()]

    # Validate sequences
    valid_sequences = []
    invalid_sequences = []

    for line in lines:
        if validate_sequence(line):
            valid_sequences.append(line)
        else:
            invalid_sequences.append(line)

    if not valid_sequences:
        return "No valid sequences found in the input", None, None

    message = f"Found {len(valid_sequences)} valid sequences"
    if invalid_sequences:
        message += f", {len(invalid_sequences)} invalid sequences will be skipped"

    # Process batch prediction
    return process_batch_prediction(valid_sequences, progress)


def select_sequence_from_batch(evt: gr.SelectData, results_df):
    """Handle selecting a sequence from batch results"""
    if results_df is None or evt.index[0] >= len(results_df):
        return None, "", ""

    # Get selected sequence
    selected_seq = results_df.iloc[evt.index[0]]["Sequence"]

    # Run prediction for this sequence
    results, _ = predict_single_sequence(selected_seq)

    if not results:
        return None, "", ""

    # Generate visualization and detailed results
    plot = plot_prediction_results(results, selected_seq)
    detailed_results = format_detailed_results(results)
    colored_seq = color_sequence(selected_seq)

    return plot, colored_seq, detailed_results


def create_example_card(sequence, description):
    """Create HTML for example sequence card"""
    return f"""
        <div style='border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 8px; 
                  background-color: white; cursor: pointer; transition: all 0.2s;'
             onclick='document.querySelector("textarea[label=\'Enter amino acid sequence\']").value = "{sequence}";'
             onmouseover='this.style.boxShadow="0 4px 6px rgba(0,0,0,0.1)"; this.style.transform="translateY(-2px)"'
             onmouseout='this.style.boxShadow="none"; this.style.transform="translateY(0)"'>
            <div style='font-family: monospace; font-weight: bold; margin-bottom: 4px;'>{sequence}</div>
            <div style='font-size: 0.85em; color: #4a5568;'>{description}</div>
        </div>
        """


def create_examples_panel():
    """Create example panel"""
    html = "<div style='margin-top: 10px;'>"
    examples = [
        (EXAMPLE_SEQUENCES[0], "Antimicrobial Peptide Sequence"),
        (EXAMPLE_SEQUENCES[1], "Magainin 2 - Classic Antimicrobial Peptide"),
        (EXAMPLE_SEQUENCES[2], "Cysteine-rich Anti-viral Peptide"),
        (EXAMPLE_SEQUENCES[3], "HIV-TAT - Cell Penetrating Peptide"),
        (EXAMPLE_SEQUENCES[4], "Anti-hypertensive Peptide")
    ]

    for seq, desc in examples:
        html += create_example_card(seq, desc)

    html += "</div>"
    return html


def load_random_example():
    """Load a random example from the example sequences"""
    import random
    return random.choice(EXAMPLE_SEQUENCES)


# Create a function to generate sequence statistics
def generate_sequence_stats(sequence):
    """Generate statistics for the sequence as HTML"""
    if not sequence or not validate_sequence(sequence):
        return ""

    # Calculate amino acid composition
    composition = {}
    for aa in sequence:
        if aa in VALID_AA:
            composition[aa] = composition.get(aa, 0) + 1

    # Calculate physicochemical properties
    # Hydrophobic amino acids
    hydrophobic = ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P']
    # Polar amino acids
    polar = ['G', 'S', 'T', 'C', 'Y', 'N', 'Q']
    # Charged amino acids
    charged = ['D', 'E', 'K', 'R', 'H']
    # Acidic amino acids
    acidic = ['D', 'E']
    # Basic amino acids
    basic = ['K', 'R', 'H']

    hydrophobic_count = sum(composition.get(aa, 0) for aa in hydrophobic)
    polar_count = sum(composition.get(aa, 0) for aa in polar)
    charged_count = sum(composition.get(aa, 0) for aa in charged)
    acidic_count = sum(composition.get(aa, 0) for aa in acidic)
    basic_count = sum(composition.get(aa, 0) for aa in basic)

    total = len(sequence)

    # Create HTML output
    html = """
        <div style='background-color: #f8fafc; border-radius: 8px; padding: 16px; margin-bottom: 20px;'>
            <h3 style='margin-top: 0; color: #2d3748; font-size: 1.2em; margin-bottom: 12px;'>Sequence Statistics</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px;'>
        """

    # Add basic statistics
    html += f"""
        <div style='background-color: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='font-weight: 600; color: #4a5568; margin-bottom: 8px;'>Basic Information</div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Sequence Length:</span>
                <span style='font-weight: 600;'>{total}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Molecular Weight (Da):</span>
                <span style='font-weight: 600;'>{total * 110:.1f}</span>
            </div>
        </div>
        """

    # Add amino acid composition properties
    html += f"""
        <div style='background-color: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='font-weight: 600; color: #4a5568; margin-bottom: 8px;'>Amino Acid Composition</div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Hydrophobic:</span>
                <span style='font-weight: 600;'>{hydrophobic_count} ({hydrophobic_count / total * 100:.1f}%)</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Polar:</span>
                <span style='font-weight: 600;'>{polar_count} ({polar_count / total * 100:.1f}%)</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Charged:</span>
                <span style='font-weight: 600;'>{charged_count} ({charged_count / total * 100:.1f}%)</span>
            </div>
        </div>
        """

    # Add charge properties
    html += f"""
        <div style='background-color: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='font-weight: 600; color: #4a5568; margin-bottom: 8px;'>Charge Properties</div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Acidic Amino Acids:</span>
                <span style='font-weight: 600;'>{acidic_count} ({acidic_count / total * 100:.1f}%)</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Basic Amino Acids:</span>
                <span style='font-weight: 600;'>{basic_count} ({basic_count / total * 100:.1f}%)</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Net Charge Estimate:</span>
                <span style='font-weight: 600;'>{basic_count - acidic_count:+d}</span>
            </div>
        </div>
        """

    html += """
            </div>
        </div>
        """

    return html


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextCNN(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_SIZE,
    n_filters=FILTER_NUM,
    filter_sizes=FILTER_SIZE,
    output_dim=OUTPUT_SIZE,
    dropout=DROPOUT
)

# Try loading the model
try:
    model.load_state_dict(torch.load('model_select+TextCNN1.pth', map_location=device))
    logger.info(f"Model loaded successfully, using device: {device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.warning("Running with an uninitialized model. Predictions will be random.")

model.to(device)
model.eval()

# Create Gradio interface
with gr.Blocks(title="Multi-functional Therapeutic Peptide Prediction System",
               css="""
                  .gradio-container {
                      max-width: 1200px !important;
                      margin: 0 auto;
                  }
                  .output-image {
                      max-height: 400px !important;
                  }
                  .info-text {
                      font-size: 0.9em;
                      color: #4a5568;
                      margin-bottom: 10px;
                  }
                  .sequence-display {
                      font-family: monospace;
                      padding: 16px;
                      background-color: #f7fafc;
                      border-radius: 8px;
                      margin-bottom: 16px;
                      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                  }
                  .article-container {
                      background-color: #f8fafc;
                      padding: 20px;
                      border-radius: 8px;
                      margin-top: 24px;
                  }
                  .footer {
                      margin-top: 30px;
                      text-align: center;
                      font-size: 0.8em;
                      color: #718096;
                  }
                  .primary-button {
                      background-color: #4299e1 !important;
                  }
                  .secondary-button {
                      background-color: #a0aec0 !important;
                  }
                  """) as demo:
    gr.Markdown(
        """
        # 🧬 Multi-functional Therapeutic Peptide Prediction System
        ## 21 Functional Predictions Based on Deep Learning TextCNN Model

        This system can predict the potential functions of peptide sequences, including antimicrobial, antiviral, anticancer, and other therapeutic functions.
        The system uses a deep learning TextCNN model to analyze amino acid sequences and predict their functions.
        """
    )

    with gr.Tabs() as tabs:
        with gr.TabItem("Single Sequence Prediction 🔍"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_sequence = gr.Textbox(
                        label="Enter Amino Acid Sequence",
                        placeholder="Example: KLGRVLSA (use only standard amino acids: ACDEFGHIKLMNPQRSTVWY)",
                        lines=3
                    )

                    with gr.Row():
                        predict_btn = gr.Button("Predict Function", variant="primary")
                        random_example_btn = gr.Button("Use Example Sequence")

                    with gr.Accordion("Example Sequences", open=False):
                        examples_html = gr.HTML(create_examples_panel())

                    prediction_message = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=3):
                    with gr.Group():
                        gr.Markdown("### Sequence Analysis")
                        colored_sequence = gr.HTML(label="Sequence", elem_classes=["sequence-display"])
                        sequence_stats = gr.HTML(label="Sequence Statistics", elem_classes=["sequence-stats"])
                        prediction_plot = gr.Plot(label="Prediction Results")
                        detailed_results = gr.HTML(label="Detailed Results")

        with gr.TabItem("Batch Prediction 📊"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Tab("File Upload"):
                        file_input = gr.File(label="Upload Sequence File (.txt, .csv or .xlsx)")
                        gr.Markdown(
                            """
                            <div class="info-text">
                            <p><strong>Notes:</strong></p>
                            <ul>
                                <li>Text File: One sequence per line</li>
                                <li>CSV/Excel File: Sequences should be in the first column, may contain headers</li>
                                <li>All sequences must only contain standard amino acids (ACDEFGHIKLMNPQRSTVWY)</li>
                                <li>The system will automatically skip invalid sequences</li>
                            </ul>
                            </div>
                            """
                        )
                        file_predict_btn = gr.Button("Process File", variant="primary")

                    with gr.Tab("Text Input"):
                        text_input = gr.Textbox(
                            label="Enter Sequences (one per line)",
                            placeholder="KLGRVLSA\nMGSTLPRTG\nFAESLKLRP",
                            lines=8
                        )
                        text_predict_btn = gr.Button("Process Sequences", variant="primary")

                    batch_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=8):
                    batch_results = gr.DataFrame(
                        headers=["Sequence", "Sequence Length", "Primary Function Prediction", "Prob",
                                 "Secondary Function Prediction", "Secondary Prob"],
                        label="Results Summary (Click rows for details)",
                        interactive=False,
                        # Add CSS to reduce font size
                        elem_classes="small-font"
                    )

                    with gr.Group():
                        gr.Markdown("### Selected Sequence Details")
                        with gr.Row():
                            with gr.Column():
                                batch_plot = gr.Plot(label="Prediction Chart")
                                batch_colored_sequence = gr.HTML(label="Selected Sequence")

                    batch_detailed_results = gr.HTML(label="Detailed Results")
                    csv_output = gr.DataFrame(visible=False)
                    download_btn = gr.Button("Download Full Results as CSV", variant="primary")
                    download_file = gr.File(label="Download Results")

        with gr.TabItem("Functional Category Information ℹ️"):
            gr.Markdown("## Peptide Functional Categories")
            gr.Markdown(
                "This system can predict 21 different peptide functional categories. The table below provides detailed information, abbreviations, and descriptions for each category.")
            gr.HTML(create_category_table())

    with gr.Accordion("About the System", open=False):
        gr.Markdown(
            """
            ### Multi-functional Therapeutic Peptide Prediction System

            This system uses the deep learning model TextCNN to predict 21 different functions of peptides. The model is trained on a large dataset of peptides with known functions, enabling it to identify potential functional patterns from amino acid sequences.

            #### Usage:

            1. **Single Sequence Prediction**: Enter a single amino acid sequence and obtain detailed functional prediction results
            2. **Batch Prediction**: Upload a file containing multiple sequences or directly input multiple lines of sequences for batch prediction
            3. **Functional Categories**: View detailed information on the 21 peptide functional categories supported by the system

            #### Technical Details:

            - Uses the TextCNN deep learning model for sequence feature extraction and functional prediction
            - Supports the standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY)
            - Prediction probability range: 0-100%, higher values indicate greater likelihood of the sequence exhibiting the respective function
            - Recommended sequence length: 5-50 amino acids (longer sequences can be processed, but the maximum effective length is 50)
            """
        )

    gr.Markdown(
        """
        <div class="footer">
        © 2025 Multi-functional Therapeutic Peptide Prediction System | Based on the TextCNN deep learning model<br>
        Last updated: March 2025 | Contact: ljj587952023@163.com
        </div>
        """
    )


    # Set up event handlers
    def update_sequence_stats(sequence):
        """Update sequence statistics"""
        if sequence and validate_sequence(sequence):
            return generate_sequence_stats(sequence)
        return ""


    # Random example button event
    random_example_btn.click(
        load_random_example,
        inputs=[],
        outputs=[input_sequence]
    )

    # Single sequence prediction event
    predict_btn.click(
        predict_tab,
        inputs=[input_sequence],
        outputs=[prediction_plot, prediction_message, colored_sequence, detailed_results]
    ).then(
        update_sequence_stats,
        inputs=[input_sequence],
        outputs=[sequence_stats]
    )

    # Batch prediction event
    file_predict_btn.click(
        batch_tab_file,
        inputs=[file_input],
        outputs=[batch_status, batch_results, csv_output]
    )

    text_predict_btn.click(
        batch_tab_text,
        inputs=[text_input],
        outputs=[batch_status, batch_results, csv_output]
    )

    # Select sequence from batch results event
    batch_results.select(
        select_sequence_from_batch,
        inputs=[batch_results],
        outputs=[batch_plot, batch_colored_sequence, batch_detailed_results]
    )


    # CSV download processing
    def create_csv_for_download(df):
        """Create CSV file for download"""
        if df is None:
            return None

        # Create CSV file from DataFrame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"peptide_predictions_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        return csv_file


    download_btn.click(
        create_csv_for_download,
        inputs=[csv_output],
        outputs=[download_file]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)


