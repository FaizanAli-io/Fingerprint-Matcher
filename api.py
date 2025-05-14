import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from image_utils import (
    remove_background,
    insert_descriptors_to_db,
    get_all_feature_names,
    find_best_matches,
    SiftFeature,
    Session,
)

app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files or "name" not in request.form:
        return jsonify({"error": "Missing image or name field"}), 400

    file = request.files["file"]
    name = request.form["name"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    os.makedirs("inputs", exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    input_filename = f"{name}_{timestamp}.jpg"
    input_path = os.path.join("inputs", input_filename)

    cv2.imwrite(input_path, img)

    processed = remove_background(img)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is not None:
        insert_descriptors_to_db(descriptors, name)

        output_img = cv2.drawKeypoints(processed, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        processed_filename = f"{name}_processed_{timestamp}.jpg"
        processed_path = os.path.join("processed", processed_filename)
        cv2.imwrite(processed_path, output_img)

        return jsonify({
            "status": "success",
            "input_image": input_filename,
            "processed_image": processed_filename
        }), 200

    return jsonify({"error": "No SIFT features found"}), 422

@app.route("/list", methods=["GET"])
def list_images():
    names = get_all_feature_names()
    print(names)
    return jsonify({"images": names})


@app.route("/match", methods=["POST"])
def match_image():
    if "file" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    os.makedirs("test", exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    input_filename = f"input_{timestamp}.jpg"
    processed_filename = f"sift_{timestamp}.jpg"
    input_path = os.path.join("test", input_filename)
    processed_path = os.path.join("test", processed_filename)

    cv2.imwrite(input_path, img)

    processed = remove_background(img)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        return jsonify({"error": "No SIFT features found"}), 422

    output_img = cv2.drawKeypoints(
        processed, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(processed_path, output_img)

    matches = find_best_matches(descriptors)
    
    print(f"Matches: {matches}")
    
    return jsonify({
        "matches": [{"name": name, "similarity": float(sim)} for name, sim in matches],
        "saved": {
            "input_image": input_filename,
            "sift_visualization": processed_filename
        }
    })


@app.route("/delete/<int:id>", methods=["DELETE"])
def delete_image(id):
    with Session() as session:
        image = session.get(SiftFeature, id)
        if not image:
            return jsonify({"error": "Image not found"}), 404
        session.delete(image)
        session.commit()
    return jsonify({"message": f"Image with ID {id} deleted"}), 204


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
