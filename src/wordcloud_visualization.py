import matplotlib.pyplot as plt

from wordcloud import WordCloud

from data_loader import load_data
from preprocess import clean_text


def generate_wordcloud():

    df = load_data("data/spam.csv")

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Spam words
    spam_words = " ".join(
        df[df['label'] == 1]['text']
    )

    # Ham words
    ham_words = " ".join(
        df[df['label'] == 0]['text']
    )

    # =====================================
    # SPAM WORD CLOUD
    # =====================================

    spam_cloud = WordCloud(
        width=800,
        height=400,
        background_color='black'
    ).generate(spam_words)

    plt.figure(figsize=(10, 5))

    plt.imshow(spam_cloud)

    plt.axis('off')

    plt.title("Spam Message Word Cloud")

    plt.show()

    # =====================================
    # HAM WORD CLOUD
    # =====================================

    ham_cloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(ham_words)

    plt.figure(figsize=(10, 5))

    plt.imshow(ham_cloud)

    plt.axis('off')

    plt.title("Ham Message Word Cloud")

    plt.show()


if __name__ == "__main__":

    generate_wordcloud()