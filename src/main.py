import os

from matcher import FingerprintMatcher


def main():
    # Directory containing fingerprint images
    data_dir = "raw_data"
    test_dir = "test_data"  # Directory for test images

    # Create fingerprint matcher
    matcher = FingerprintMatcher(data_dir)

    # Create output directory for visualizations
    os.makedirs("output", exist_ok=True)

    # Visualize SIFT features for each fingerprint
    print("\nVisualizing SIFT features for each fingerprint:")
    for person_id in matcher.fingerprint_db:
        image_path = os.path.join(data_dir, f"{person_id}.png")  # Assuming PNG format
        if not os.path.exists(image_path):
            # Try other formats
            for ext in [".jpg", ".jpeg", ".bmp"]:
                image_path = os.path.join(data_dir, f"{person_id}{ext}")
                if os.path.exists(image_path):
                    break

        save_path = os.path.join("output", f"{person_id}_features.png")
        matcher.visualize_features(image_path, save_path)

    print("\nTesting fingerprint matching:")

    # Loop through all images in test_data directory
    for test_image_name in os.listdir(test_dir):
        test_image = os.path.join(test_dir, test_image_name)
        if not os.path.isfile(test_image):
            continue  # Skip if it's not a file (e.g., subdirectory)

        print(f"Using test image: {test_image}")

        best_match, score, num_good_matches = matcher.match(test_image)
        print(
            f"Best match: {best_match} with score: {score:.4f} ({num_good_matches} good matches)"
        )

        if best_match:
            # Visualize matches
            save_path = os.path.join(
                "output", f"match_visualization_{test_image_name}.png"
            )
            matcher.visualize_matches(test_image, best_match, save_path)


if __name__ == "__main__":
    main()
