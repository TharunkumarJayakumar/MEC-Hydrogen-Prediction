import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

# Load dataset
df = pd.read_csv("data/literature_dataset.csv", encoding="latin1")

# Drop extra unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("Original Dataset Shape:", df.shape)
print("Columns:", df.columns)

# Create metadata
metadata = Metadata.detect_from_dataframe(data=df)

# Initialize CTGAN synthesizer
model = CTGANSynthesizer(metadata, epochs=500)

# Fit model
model.fit(df)

# Generate synthetic data
synthetic_data = model.sample(num_rows=500)

print("\nSynthetic Dataset Shape:", synthetic_data.shape)
print(synthetic_data.head())

# Save synthetic dataset
synthetic_data.to_csv("synthetic_dataset.csv", index=False)
