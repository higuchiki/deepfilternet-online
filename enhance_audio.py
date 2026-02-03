import os
import argparse
from df.enhance import enhance, init_df, load_audio, save_audio

def main():
    parser = argparse.ArgumentParser(description="DeepFilterNet Audio Enhancement")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output audio file (optional)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_enhanced" + ext

    print(f"Initializing DeepFilterNet...")
    model, df_state, _ = init_df()

    print(f"Loading audio: {input_path}")
    audio, info = load_audio(input_path, sr=df_state.sr())

    print(f"Enhancing audio...")
    enhanced = enhance(model, df_state, audio)

    print(f"Saving enhanced audio: {output_path}")
    save_audio(output_path, enhanced, sr=df_state.sr())
    print("Done!")

if __name__ == "__main__":
    main()
