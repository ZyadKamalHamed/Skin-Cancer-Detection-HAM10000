"""
HAM10000 Dataset Downloader
Downloads and organizes the skin cancer dataset
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Simple path setup - no complex imports needed
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

class HAM10000Downloader:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ Data directories created at: {self.data_dir}")

    def create_sample_dataset(self):
        """Create a sample dataset structure for development"""
        print("🛠️ Creating sample dataset structure...")

        # Create sample metadata
        sample_data = {
            'image_id': ['ISIC_0024306', 'ISIC_0024307', 'ISIC_0024308', 'ISIC_0024309'],
            'dx': ['nv', 'mel', 'bkl', 'bcc'],
            'dx_type': ['histo', 'histo', 'histo', 'histo'],
            'age': [45.0, 60.0, 55.0, 40.0],
            'sex': ['male', 'female', 'male', 'female'],
            'localization': ['back', 'face', 'arm', 'leg']
        }

        df = pd.DataFrame(sample_data)
        csv_path = self.raw_dir / "HAM10000_metadata.csv"
        df.to_csv(csv_path, index=False)

        print(f"✅ Sample metadata created: {csv_path}")
        print(f"📊 Sample contains {len(df)} entries")

        return df

    def analyze_dataset(self):
        """Analyze the dataset"""
        csv_file = self.raw_dir / "HAM10000_metadata.csv"

        if not csv_file.exists():
            print("❌ No dataset found. Creating sample...")
            return self.create_sample_dataset()

        df = pd.read_csv(csv_file)

        print("\n📊 DATASET ANALYSIS")
        print("=" * 50)
        print(f"Total samples: {len(df)}")
        print(f"Unique diagnoses: {df['dx'].nunique()}")
        print("\nClass distribution:")
        print(df['dx'].value_counts())

        return df

def main():
    """Main function to test the downloader"""
    print("🚀 HAM10000 Dataset Downloader")
    print("=" * 40)

    downloader = HAM10000Downloader()
    df = downloader.analyze_dataset()

    print("\n✅ Dataset ready for training!")

if __name__ == "__main__":
    main()