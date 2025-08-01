import os
import csv
import argparse

def adjust_precision(logical_error_rate, num_samples):
    """Round logical_error_rate to the nearest multiple of 1/num_samples."""
    try:
        num_samples = int(num_samples)
        logical_error_rate = float(logical_error_rate)
        unit = 1 / num_samples
        rounded = round(logical_error_rate / unit) * unit
        return f"{rounded:.3f}"  # Keeps 6 decimal digits for consistency
    except ValueError:
        return logical_error_rate  # Leave unchanged if not a number

def process_csv(filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return

    header = rows[0]
    try:
        num_samples_idx = header.index("num_samples")
        ler_idx = header.index("logical_error_rate")
    except ValueError:
        return  # Skip files without the right headers
    # Adjust the logical_error_rate in all rows
    changed = False
    for i in range(1, len(rows)):
        row = rows[i]
        if len(row) <= max(num_samples_idx, ler_idx):
            continue  # Skip malformed rows

        old_val = row[ler_idx]
        new_val = adjust_precision(old_val, row[num_samples_idx])

        if new_val != old_val:
            row[ler_idx] = new_val
            changed = True

    if changed:
        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Updated: {filepath}")

def process_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".csv"):
                process_csv(os.path.join(dirpath, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix logical_error_rate precision in CSV files.")
    parser.add_argument("path", help="Path to the root directory")
    args = parser.parse_args()

    process_directory(args.path)