import cv2
import numpy as np
from PIL import Image
from rembg import remove
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, Text, Float

# --- SQLAlchemy Setup ---
DB_URL = "postgresql://neondb_owner:npg_iSgrV0csFlw2@ep-shrill-frog-a47kkrww-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class SiftFeature(Base):
    __tablename__ = "sift_features"
    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    descriptor = Column(ARRAY(Float), nullable=False)


Base.metadata.create_all(engine)


# --- Image Processing Functions ---
def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def remove_background(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    result = remove(pil_img)
    result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    if result.mode == "RGBA":
        alpha = np.array(result.split()[-1])
        mask = (alpha > 0).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(result_cv, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = result_cv[y : y + h // 2, x : x + w]
    resized = cv2.resize(cropped, (256, 256))
    return enhance_contrast(resized)


def extract_sift_descriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


# --- DB Functions ---
def insert_descriptors_to_db(descriptors, name):
    if descriptors is None:
        return
    flat = descriptors.flatten().tolist()
    with Session() as session:
        entry = SiftFeature(name=name, descriptor=flat)
        session.add(entry)
        session.commit()


def get_all_feature_names():
    with Session() as session:
        return [(row.id, row.name) for row in session.query(SiftFeature).all()]


def find_best_matches(descriptors, top_k=5):
    if descriptors is None:
        return []

    flat = descriptors.flatten().reshape(1, -1)
    with Session() as session:
        all_entries = session.query(SiftFeature).all()
        scores = []
        for entry in all_entries:
            stored = np.array(entry.descriptor).reshape(1, -1)
            min_len = min(flat.shape[1], stored.shape[1])
            sim = cosine_similarity(flat[:, :min_len], stored[:, :min_len])[0][0]
            scores.append((entry.name, sim))
        return sorted(scores, key=lambda x: -x[1])[:top_k]
