import os
import pandas as pd

# Paths
folder1 = "/home/sahil/Desktop/Rl_Project_Final/data/Metadata"
folder2 = "/home/sahil/Desktop/Rl_Project_Final/data/metadata_xls"
output_folder = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Function to convert excel to csv
def convert_folder_to_csv(folder_path):
    data_dict = {}
    
    for file in os.listdir(folder_path):
        if file.endswith(".xls") or file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)
            
            try:
                df = pd.read_excel(file_path)
                
                csv_name = os.path.splitext(file)[0] + ".csv"
                csv_path = os.path.join(output_folder, csv_name)
                
                df.to_csv(csv_path, index=False)
                print(f"✅ Converted: {file} → {csv_name}")
                
                data_dict[file] = df
                
            except Exception as e:
                print(f"❌ Error in {file}: {e}")
    
    return data_dict

print("\n--- Converting Folder 1 ---")
data1 = convert_folder_to_csv(folder1)

print("\n--- Converting Folder 2 ---")
data2 = convert_folder_to_csv(folder2)

# 🔍 Compare files
print("\n--- Comparing Data ---")

common_files = set(data1.keys()).intersection(set(data2.keys()))

for file in common_files:
    try:
        same = data1[file].equals(data2[file])
        print(f"{file}: {'✅ SAME' if same else '❌ DIFFERENT'}")
    except:
        print(f"{file}: ⚠️ Could not compare")

# Check if both folders have same filenames
if set(data1.keys()) == set(data2.keys()):
    print("\n📁 Both folders contain SAME file names")
else:
    print("\n📁 Folders have DIFFERENT file sets")

import os
import pandas as pd

csv_folder = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv"

print(f"\n📂 Reading CSV files from: {csv_folder}\n")

for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder, file)

        try:
            df = pd.read_csv(file_path)

            print(f"\n📄 File: {file}")
            print("-" * 60)
            print("Shape:", df.shape)
            print("Columns:", list(df.columns))
            print(df.head())   # 👈 first 5 rows

        except Exception as e:
            print(f"❌ Error in {file}: {e}")

print("\n🎯 DONE")


import pandas as pd

# ==============================
# 📌 LOAD FILES
# ==============================
video_path = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv/video_list_fixed.csv"
ratings_path = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv/participant_ratings.csv"

video_df = pd.read_csv(video_path)
ratings_df = pd.read_csv(ratings_path)

# ==============================
# 🧹 CLEAN DATA
# ==============================

# Remove NaN Experiment_id from video list
video_df = video_df.dropna(subset=["Experiment_id"])

# Convert to int for safe join
video_df["Experiment_id"] = video_df["Experiment_id"].astype(int)
ratings_df["Experiment_id"] = ratings_df["Experiment_id"].astype(int)

# ==============================
# 📊 AGGREGATE RATINGS
# ==============================
# Average Valence & Arousal per Experiment_id

ratings_agg = (
    ratings_df
    .groupby("Experiment_id")[["Valence", "Arousal"]]
    .mean()
    .reset_index()
)

# ==============================
# 🔗 MERGE DATA
# ==============================
final_df = pd.merge(
    video_df[["Experiment_id", "Lastfm_tag"]],
    ratings_agg,
    on="Experiment_id",
    how="inner"
)

# Rename column
final_df.rename(columns={"Lastfm_tag": "Emotion"}, inplace=True)

# ==============================
# 📌 RESULT
# ==============================
print("✅ Final DataFrame:")
print(final_df.head())

print("\nShape:", final_df.shape)


import pandas as pd

# ==============================
# 📌 LOAD FINAL MERGED DATA
# ==============================
video_path = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv/video_list_fixed.csv"
ratings_path = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv/participant_ratings.csv"

video_df = pd.read_csv(video_path)
ratings_df = pd.read_csv(ratings_path)

# ==============================
# 🧹 CLEAN
# ==============================
video_df = video_df.dropna(subset=["Experiment_id"])
video_df["Experiment_id"] = video_df["Experiment_id"].astype(int)
ratings_df["Experiment_id"] = ratings_df["Experiment_id"].astype(int)

# ==============================
# 📊 STEP 1: AVG per Experiment
# ==============================
ratings_agg = (
    ratings_df
    .groupby("Experiment_id")[["Valence", "Arousal"]]
    .mean()
    .reset_index()
)

# ==============================
# 🔗 STEP 2: MERGE WITH EMOTION
# ==============================
merged_df = pd.merge(
    video_df[["Experiment_id", "Lastfm_tag"]],
    ratings_agg,
    on="Experiment_id",
    how="inner"
)

merged_df.rename(columns={"Lastfm_tag": "Emotion"}, inplace=True)

# ==============================
# 📊 STEP 3: AVG per Emotion
# ==============================
emotion_avg = (
    merged_df
    .groupby("Emotion")[["Valence", "Arousal"]]
    .mean()
    .reset_index()
)

# ==============================
# 💾 SAVE CSV
# ==============================
output_path = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv/emotion_summary.csv"
emotion_avg.to_csv(output_path, index=False)

# ==============================
# 📌 OUTPUT
# ==============================
print("✅ Emotion-wise Average:")
print(emotion_avg)

print(f"\n💾 Saved at: {output_path}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 📌 LOAD DATA
# ==============================
path = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv/emotion_summary.csv"
df = pd.read_csv(path)

# ==============================
# 🧹 NORMALIZE TO [-1, 1]
# ==============================

def normalize(col):
    return 2 * (col - col.min()) / (col.max() - col.min()) - 1

df["Valence_norm"] = normalize(df["Valence"])
df["Arousal_norm"] = normalize(df["Arousal"])

# ==============================
# 🔵 SCALE TO UNIT CIRCLE
# ==============================
r = np.sqrt(df["Valence_norm"]**2 + df["Arousal_norm"]**2)
max_r = r.max()

df["Valence_unit"] = df["Valence_norm"] / max_r
df["Arousal_unit"] = df["Arousal_norm"] / max_r

# ==============================
# 🎨 PLOT
# ==============================
plt.figure(figsize=(8,8))

# Draw unit circle
circle = plt.Circle((0, 0), 1, fill=False)
plt.gca().add_patch(circle)

# Scatter points
plt.scatter(df["Valence_unit"], df["Arousal_unit"])

# Annotate emotions
for i in range(len(df)):
    plt.text(
        df["Valence_unit"][i],
        df["Arousal_unit"][i],
        df["Emotion"][i],
        fontsize=9
    )

# Axes
plt.axhline(0)
plt.axvline(0)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.title("Emotion Wheel (Normalized Unit Circle)")

plt.grid()
plt.gca().set_aspect('equal', adjustable='box')

plt.show()


def normalize_va(v, a):
    v_norm = 2 * (v - 1) / (9 - 1) - 1
    a_norm = 2 * (a - 1) / (9 - 1) - 1
    return v_norm, a_norm


    