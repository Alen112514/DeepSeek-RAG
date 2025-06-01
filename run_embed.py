from data_embedding import init_text_chroma, init_image_chroma

if __name__ == "__main__":
    print("Embedding text from EPUB...")
    init_text_chroma()
    print("✅ Text embedded and saved.")

    print("Embedding images from EPUB...")
    init_image_chroma()
    print("✅ Images embedded and saved.")
