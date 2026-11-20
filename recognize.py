"""Enroll reference faces and recognize enrolled identities in images."""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.nn.functional import normalize

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class FaceRecognizer:
    def __init__(self):
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = MTCNN(
            image_size=160, keep_all=True, min_face_size=40, device=self.device
        ).eval()
        self.encoder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.names, self.embeddings = [], None

    @torch.inference_mode()
    def encode(self, image):
        image = image.convert("RGB")
        boxes, probabilities = self.detector.detect(image)
        if boxes is None:
            return [], torch.empty(0, 512)
        selected = [
            index
            for index, probability in enumerate(probabilities)
            if probability >= 0.9
        ]
        if not selected:
            return [], torch.empty(0, 512)
        boxes = boxes[selected]
        faces = self.detector.extract(image, boxes, save_path=None)
        embeddings = normalize(self.encoder(faces.to(self.device)), dim=1).cpu()
        return boxes.tolist(), embeddings

    def enroll(self, images, output):
        if output.exists():
            raise ValueError("Gallery already exists. Choose another output path.")
        identities = sorted(path for path in images.iterdir() if path.is_dir())
        names, averages = [], []
        for identity in identities:
            collected = []
            for path in sorted(identity.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                with Image.open(path) as image:
                    _, embeddings = self.encode(image)
                if len(embeddings) == 1:
                    collected.append(embeddings[0])
                else:
                    print(f"Skipped {path.name}: expected one confident face.")
            if not collected:
                raise ValueError(f"No usable reference face for {identity.name}.")
            names.append(identity.name)
            averages.append(normalize(torch.stack(collected).mean(0), dim=0))
        if not names:
            raise ValueError("Provide one reference-image folder per identity.")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"format_version": 1, "names": names, "embeddings": torch.stack(averages)},
            output,
        )
        print(f"Saved {len(names)} enrolled identities to {output}")

    def load_gallery(self, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        names, embeddings = data["names"], data["embeddings"]
        if (
            data.get("format_version") != 1
            or not names
            or len(set(names)) != len(names)
        ):
            raise ValueError("Invalid face gallery.")
        if (
            embeddings.shape != (len(names), 512)
            or not torch.isfinite(embeddings).all()
        ):
            raise ValueError("Gallery embeddings must have shape [identities, 512].")
        if (embeddings.norm(dim=1) < 1e-8).any():
            raise ValueError("Gallery embeddings cannot be zero vectors.")
        self.names, self.embeddings = names, normalize(embeddings.float(), dim=1)

    def recognize(self, image, threshold=0.9):
        if not math.isfinite(threshold) or not 0 < threshold <= 2:
            raise ValueError("Distance threshold must be in (0, 2].")
        if self.embeddings is None:
            raise ValueError("Load a gallery before recognition.")
        boxes, embeddings = self.encode(image)
        records = []
        for box, embedding in zip(boxes, embeddings, strict=True):
            distances = torch.cdist(embedding.unsqueeze(0), self.embeddings)[0]
            distance, index = distances.min(0)
            name = (
                self.names[index.item()] if distance.item() <= threshold else "unknown"
            )
            records.append({"name": name, "distance": distance.item(), "box": box})
        return records


def annotate(image, records):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for record in records:
        color = "red" if record["name"] == "unknown" else "lime"
        draw.rectangle(record["box"], outline=color, width=3)
        x, y = record["box"][:2]
        draw.text(
            (x, max(0, y - 14)),
            f"{record['name']} ({record['distance']:.2f})",
            fill=color,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    enroll = commands.add_parser("enroll")
    enroll.add_argument("--images", type=Path, required=True)
    enroll.add_argument("--output", type=Path, default=Path("embeddings.pt"))
    detect = commands.add_parser("recognize")
    detect.add_argument("--gallery", type=Path, default=Path("embeddings.pt"))
    detect.add_argument("--image", type=Path, required=True)
    detect.add_argument("--output", type=Path, default=Path("outputs/recognized.png"))
    detect.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    try:
        engine = FaceRecognizer()
        if args.command == "enroll":
            engine.enroll(args.images, args.output)
        else:
            if args.output.exists():
                raise ValueError("Output already exists. Choose another filename.")
            engine.load_gallery(args.gallery)
            with Image.open(args.image) as image:
                records = engine.recognize(image, args.threshold)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                annotate(image, records).save(args.output)
            print(json.dumps(records, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
