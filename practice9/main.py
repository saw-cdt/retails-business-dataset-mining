import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/data/processed/data_clean.csv")

cat_cols = [
    "Region",
    "Product_Category",
    "Customer_Segment",
    "Payment_Method"
]

print("=" * 60)
print("TEXT ANALYSIS — WORD CLOUD")
print("=" * 60)

# Generate word cloud per categorical column
for col in cat_cols:
    counts = df[col].value_counts().to_dict()
    
    plt.figure(figsize=(10, 6))
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        random_state=42
    ).generate_from_frequencies(counts)
    
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — {col}", fontsize=14)
    plt.tight_layout(pad=0)
    plt.savefig(f"/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/practice9/wordcloud_{col.lower()}.png", dpi=150)
    print(f"Saved: wordcloud_{col.lower()}.png")
    plt.show()

# Generate combined word cloud from all categorical values
all_text = ""
for col in cat_cols:
    all_text += " ".join(df[col].astype(str).tolist()) + " "

plt.figure(figsize=(12, 6))
wc_all = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    colormap="plasma",
    random_state=42
).generate(all_text)

plt.imshow(wc_all, interpolation="bilinear")
plt.axis("off")
plt.title("Combined Word Cloud — All Categories", fontsize=14)
plt.tight_layout(pad=0)
plt.savefig("/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/practice9/wordcloud_combined.png", dpi=150)
print("Saved: wordcloud_combined.png")
plt.show()

print("\n" + "=" * 60)
print("WORD CLOUD GENERATION COMPLETE")
print("=" * 60)
