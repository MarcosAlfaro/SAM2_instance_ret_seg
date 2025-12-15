### INSTANCE SEGMENTATION

Current approach:
Create a pseudo-video with the reference/support image and the target image
Use the available support mask as prompt 
Use the mask obtained from frozen SAM2 as output


# PerSeg
1. set --dataset_file to perseg
2. run sam2_permis.py

# PerMIS
1. set --dataset_file to perseg
2. run sam2_permis.py

### INSTANCE RETRIEVAL

Current approach: 
Use the IoU prediction score as a probability score that the object is present in the image
This score works without training but can be improved

# Pipeline 1 (wo Training)

1. change --dataset_file to permir
2. run sam2_permir_obj_scores.py twice (once to load images/masks/labels onto a .pt file and another for retrieval)

# PerMIR

# Pipeline 2 (Training with R-Oxford & Paris, Evaluation with ILIAS)

# Training (R-Oxford & R-Paris)
- 10 epochs, 1k samples/ep, Loss: Lazy Triplet (m=0.5)
- Triplet sample building: random choice, except for anchor must belong to the query set, and cls(anc)==cls(pos)!=cls(neg)

# TODO: Implement the UnED dataset
UnED dataset is composed of 8 subdatasets: 
    CUB-200-2011 (Birds): bounding boxes and masks
    CARS196 (Stanford cars): bounding boxes
    DeepFashion (Clothes): bounding boxes *Requires permission and license agreement
    RP2k (Retail products): bounding boxes *Only in the detection split
    Google Landmarks v2 (Landmarks): bounding boxes *Should be in the Google Landmarks Detection set from Kaggle, but I didn’t find them
    Stanford Online Products: only class labels
    Food2k and Food-101: only class labels
    Met (Art): only class labels
*We need to decide which should we use
* We could use all of them and for the ones wo bounding boxes we can prompt the entire image

# Evaluation (ILIAS)
Two-stage retrieval:
1st stage: DINOv2-Small (cls token) -> top-100 candidates 
# Future experiments -> extend to 1k
2nd stage: SAM2-Tiny (IoU pred score)
# Future experiments -> extend to SAM2-L

Current shortcut to balance inference time and storage:
For the 1st stage: Run the code online with huggingface.datasets
For the 2nd stage: Download every distractor image that appears within the top-K candidates 

How to run
1. precompute_features_ilias.py
    - Extract the features from every image of the ILIAS dataset with DINOv2 (queries, db and distractors)
    - Features are stored in .npy files
2. topk_candidates_ilias.py
    - similarity search with faiss idx
    - topk indices for every query are stored in .npy files
3. extract_images.py
    - download queries and database images
    - run only once
4. extract_candidates.py
    - download distractors that appear within the topk candidates of at least one query
    - run every time we change of backbone for the 1st stage
    # TODO: download all the images and work with them locally
5. train_revisitop.py
    - train the IoU prediction head from the SAM2 model
    # TODO: replace with the UnED dataset
6. sam2_reranking.py
    - perform re-ranking with the IoU score
    - results are stored in .npy files and displayed in the terminal




