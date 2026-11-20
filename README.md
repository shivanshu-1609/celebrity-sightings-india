# Celebrity Sightings in India

Face detection and recognition using MTCNN and InceptionResnetV1, with a Flask
image-upload API. Recognition compares detected faces with an explicitly enrolled
gallery rather than identifying arbitrary people.

## Data and models

The course uses public-figure interview material and reference images, including
Mary Kom. Supply authorized reference images in one folder per identity:

```text
references/
├── mary_kom/
│   ├── image_01.jpg
│   └── image_02.jpg
└── ranveer/
    └── image_01.jpg
```

Only reference images containing one confident detected face are enrolled. Raw
reference images and embedding files are excluded from version control.
MTCNN crops 160 × 160 faces; InceptionResnetV1 uses pretrained VGGFace2 weights
and produces 512-dimensional embeddings. Weights download on first use.

## Setup and usage

Use Python 3.11 and a separate virtual environment; this project's FaceNet release
uses the PyTorch 2.2/torchvision 0.17 dependency pair.

```bash
python -m pip install -r requirements.txt
python recognize.py enroll --images references --output embeddings.pt
python recognize.py recognize --gallery embeddings.pt --image data/example.jpg
python app.py --gallery embeddings.pt
```

Open `http://127.0.0.1:5000` to upload an image. The API is also available directly:

```bash
curl -F "image=@data/example.jpg" http://127.0.0.1:5000/recognize
```

The response contains names, embedding distances, and bounding boxes. The CLI also
saves an annotated image. Empty detections return an empty list. Distances above
`--threshold` are labeled `unknown`; distance is not an identity probability.

## Evaluation

The default distance threshold of 0.9 is a starting setting, not a measured accuracy
claim. Evaluate it on separate reference and test images, including unknown people.
Lighting, pose, low-resolution faces, and lookalikes can affect matching. No new
recognition benchmark is claimed from the supplied materials.

## References and license

[WorldQuant University](https://www.wqu.edu/) provided the course notebooks.
[facenet-pytorch](https://github.com/timesler/facenet-pytorch) provides the detector
and embedding model. Standalone scripts use MIT; original course material,
pretrained weights, and reference images retain their own terms.
