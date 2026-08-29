from src.data_loader import load_full_dataset

if __name__ == "__main__":
    df = load_full_dataset("data/raw")
    print("\n--- PHASE 2 SUCCESSFUL ---")
    print(f"Final Merged Shape: {df.shape}")
    