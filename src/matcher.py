import os
import cv2
from matplotlib import pyplot as plt


class FingerprintMatcher:
    def __init__(self, data_dir):
        """
        Initialize the fingerprint matcher with the directory containing raw fingerprint images.

        Args:
            data_dir (str): Directory path containing fingerprint images
        """
        self.data_dir = data_dir
        self.fingerprint_db = (
            {}
        )  # Dictionary to store processed fingerprints and their keypoints/descriptors

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()

        # FLANN parameters for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Load fingerprint database
        self.load_fingerprints()

    def load_fingerprints(self):
        """Load all fingerprint images from the data directory and extract SIFT features."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        print(f"Loading fingerprints from {self.data_dir}")
        for filename in os.listdir(self.data_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                filepath = os.path.join(self.data_dir, filename)
                print(f"Processing {filename}")

                # Use filename without extension as the person's identifier
                person_id = os.path.splitext(filename)[0]

                # Process the fingerprint
                processed_img = self.preprocess_image(filepath)
                keypoints, descriptors = self.extract_sift_features(processed_img)

                # Store in database
                self.fingerprint_db[person_id] = {
                    "processed_image": processed_img,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                }

        print(f"Loaded {len(self.fingerprint_db)} fingerprints")

    def preprocess_image(self, image_path):
        """
        Preprocess the fingerprint image to enhance features.

        Args:
            image_path (str): Path to the fingerprint image

        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Resize to standardize image size (optional)
        img = cv2.resize(img, (500, 500))

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)

        return img

    def extract_sift_features(self, img):
        """
        Extract SIFT features from a preprocessed image.

        Args:
            img (numpy.ndarray): Preprocessed fingerprint image

        Returns:
            tuple: (keypoints, descriptors)
        """
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(img, None)

        return keypoints, descriptors

    def match(self, query_image_path, threshold=0.7, min_matches=10):
        """
        Match a query fingerprint against the database using SIFT features.

        Args:
            query_image_path (str): Path to the query fingerprint image
            threshold (float): Ratio threshold for Lowe's ratio test
            min_matches (int): Minimum number of good matches required

        Returns:
            tuple: (best_match_id, score, num_good_matches)
        """
        # Preprocess the query image
        processed_query = self.preprocess_image(query_image_path)

        # Extract SIFT features from query image
        query_keypoints, query_descriptors = self.extract_sift_features(processed_query)

        best_match = None
        highest_score = 0
        best_matches = None
        highest_matches_count = 0

        # Compare against each fingerprint in the database
        for person_id, data in self.fingerprint_db.items():
            db_keypoints = data["keypoints"]
            db_descriptors = data["descriptors"]

            if query_descriptors is None or db_descriptors is None:
                continue

            if len(query_descriptors) == 0 or len(db_descriptors) == 0:
                continue

            # Match descriptors using FLANN
            matches = self.flann.knnMatch(query_descriptors, db_descriptors, k=2)

            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good_matches.append(m)

            # Calculate match score
            score = len(good_matches) / max(len(query_keypoints), len(db_keypoints))

            print(
                f"Match score for {person_id}: {score:.4f} (good matches: {len(good_matches)})"
            )

            if (
                len(good_matches) > highest_matches_count
                and len(good_matches) >= min_matches
            ):
                highest_matches_count = len(good_matches)
                highest_score = score
                best_match = person_id
                best_matches = good_matches

        return best_match, highest_score, highest_matches_count

    def visualize_matches(self, query_image_path, match_id, save_path=None):
        """
        Visualize matching keypoints between query image and best match.

        Args:
            query_image_path (str): Path to the query fingerprint image
            match_id (str): ID of the matching fingerprint
            save_path (str, optional): Path to save the visualization
        """
        if match_id not in self.fingerprint_db:
            print(f"No match found with ID: {match_id}")
            return

        # Preprocess query image
        processed_query = self.preprocess_image(query_image_path)
        query_keypoints, query_descriptors = self.extract_sift_features(processed_query)

        # Get data for the matched fingerprint
        match_data = self.fingerprint_db[match_id]
        match_img = match_data["processed_image"]
        match_keypoints = match_data["keypoints"]
        match_descriptors = match_data["descriptors"]

        # Match descriptors
        matches = self.flann.knnMatch(query_descriptors, match_descriptors, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Draw matches
        img_matches = cv2.drawMatches(
            processed_query,
            query_keypoints,
            match_img,
            match_keypoints,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Display
        plt.figure(figsize=(16, 8))
        plt.imshow(img_matches)
        plt.title(
            f"Matches between query and {match_id} - {len(good_matches)} good matches"
        )
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")

        plt.show()

    def visualize_features(self, image_path, save_path=None):
        """
        Visualize SIFT keypoints on a fingerprint image.

        Args:
            image_path (str): Path to the fingerprint image
            save_path (str, optional): Path to save the visualization
        """
        # Preprocess the image
        processed_img = self.preprocess_image(image_path)

        # Extract SIFT features
        keypoints, _ = self.extract_sift_features(processed_img)

        # Draw keypoints
        img_with_keypoints = cv2.drawKeypoints(
            processed_img,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        # Display
        plt.figure(figsize=(8, 8))
        plt.imshow(img_with_keypoints)
        plt.title(
            f"SIFT Features: {os.path.basename(image_path)} - {len(keypoints)} keypoints"
        )
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")

        plt.show()
