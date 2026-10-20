# Assets Directory

Place evaluation images here for use with the evaluation pipeline.

## Usage

1. Download a small subset of images from an established dataset such as:
   - **Flickr8k**: Available on [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
   - **MS COCO**: Available at [cocodataset.org](https://cocodataset.org/#download)

2. Place the image files (JPG, PNG, or WEBP) directly in this directory.

3. Update `evaluation/sample_references.json` with matching filenames and
   their human-written reference captions from the dataset annotations.

## Important

- Do **not** commit large image datasets to the Git repository.
- Verify the dataset's license permits your intended use.
- This directory is listed in `.gitignore` for image files by default;
  only this README is tracked.
