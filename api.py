import cv2
import numpy as np
from flask import Flask, request, jsonify
from image_utils import (
    remove_background,
    extract_sift_descriptors,
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

    processed = remove_background(img)
    descriptors = extract_sift_descriptors(processed)
    if descriptors is not None:
        insert_descriptors_to_db(descriptors, name)
        return jsonify({"status": "success"}), 200
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

    processed = remove_background(img)
    descriptors = extract_sift_descriptors(processed)
    matches = find_best_matches(descriptors)

    return jsonify(
        {"matches": [{"name": name, "similarity": float(sim)} for name, sim in matches]}
    )


@app.route("/delete/<int:id>", methods=["DELETE"])
def delete_image(id):
    with Session() as session:
        image = session.get(SiftFeature, id)
        if not image:
            return jsonify({"error": "Image not found"}), 404
        session.delete(image)
        session.commit()
    return jsonify({"message": f"Image with ID {id} deleted"}), 200


if __name__ == "__main__":
    app.run(port=8080)
