# Claude AI Assistant Configuration

This file contains configuration and context information for Claude AI assistant to better understand and work with this cattle recognition project.

## Project Overview
**Cattle Recognition System** - An automated system for detection and tracking of individual cattle using computer vision and machine learning techniques. The system processes video input to locate each cow and assign unique, persistent identities automatically as they are discovered.

## Development Plan - AI Model Development Process

### Objective
Develop a system that processes video to locate each cow and assign unique, persistent identities automatically.

### Phase 1: Dataset Creation
The system needs to learn in two distinct ways, requiring two types of data:

#### 1.1 Detection Data
- **Concept**: Gather a large number of images containing cattle
- **Requirement**: Each cow in each image must be clearly marked with bounding boxes
- **Purpose**: Teach the system to answer: "Where is there a cow?"

#### 1.2 Identification Data
- **Concept**: Learn visual features that distinguish individual cattle without identity labels
- **Current Data**: Multiple images of different cattle from various angles (cow-counting-v3: 3,371 images)
- **Challenge**: No ground truth identity labels available (all cows are random/unrelated)
- **Approach**: Unsupervised contrastive learning to extract distinctive visual characteristics
- **Purpose**: Train the model to generate unique "fingerprints" for cattle re-identification
- **Method**: Self-supervised learning (SimCLR/MoCo) + feature clustering for identity assignment
- **Technical Note**: Cannot use Siamese/Triplet loss due to lack of positive/negative pairs ground truth

### Phase 2: System Capability Training
Build two AI "engines":

#### 2.1 Detection Capability Training
- **Concept**: Use first dataset to train a model specialized in locating and drawing bounding boxes around all cattle in an image or video frame

#### 2.2 Differentiation Capability Training
- **Concept**: Use unsupervised learning to train a model specialized in generating distinctive embeddings
- **Architecture**: ResNet/EfficientNet backbone + contrastive learning head
- **Training Strategy**: Self-supervised contrastive learning with image augmentations as positive pairs
- **Output**: Convert cattle visual patterns into unique numerical "fingerprint" (512D feature vector)
- **Similarity Threshold**: Critical parameter requiring careful calibration during deployment

### Phase 3: Real-World Optimization
- **Concept**: Convert and compress trained models to lightweight format
- **Goal**: Ensure smooth real-time operation on devices with limited computing capacity (e.g., drones, Raspberry Pi)
- **Requirements**: Low memory and processor consumption

### Phase 4: Tracking Logic Development
Main program that uses trained models for the final task:

#### Core Loop:
1. **Initialize**: Create empty database to store known cattle "fingerprints"
2. **Capture**: Receive new video frame
3. **Detect**: Use first model to find locations of all cattle in frame
4. **For each cow found**:
   - **Extract Crop**: Extract cow region using bounding box coordinates
   - **Generate Fingerprint**: Use identification model to create 512D feature vector
   - **Similarity Search**: Calculate cosine similarity with all stored fingerprints in database
   - **Identity Decision**:
     - If max_similarity > THRESHOLD: assign existing ID with highest similarity
     - If max_similarity ≤ THRESHOLD: create new cow ID, add fingerprint to database
5. **Display**: Show cow bounding box with assigned ID on screen

### Phase 5: Field Testing and Calibration
#### 5.1 Implementation
- Install and run complete system on final device (e.g., Raspberry Pi)

#### 5.2 Validation
- Conduct field tests with real cattle to observe behavior and accuracy

#### 5.3 Fine Tuning
- Calibrate similarity threshold parameter
- **Critical**: This numerical value determines how similar two fingerprints must be to consider them from the same cow

## Development Environment
- **Language**: Python
- **Focus**: AI model development best practices
- **Target**: Real-time video processing system

## Project Structure
```
cattle-recognition/
├── data/
│   ├── detection/          # Detection training images and annotations
│   └── identification/     # Individual cattle images for ID training
├── models/
│   ├── detection/          # Detection model files
│   ├── identification/     # Identification/ReID model files
│   └── optimized/          # Compressed models for deployment
├── src/
│   ├── data_preparation/   # Dataset creation and preprocessing
│   ├── training/           # Model training scripts
│   ├── inference/          # Model inference and optimization
│   └── tracking/           # Main tracking system logic
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks for experimentation
└── deployment/             # Deployment scripts and configurations
```

## Development Commands
- `source .venv/bin/activate` - access to virtual enviroment
- `python -m pytest tests/` - Run all tests
- `python src/training/train_detection.py` - Train detection model
- `python src/training/train_identification.py` - Train identification model
- `python src/tracking/main.py` - Run complete tracking system
- `pip install -r requirements.txt` - Install dependencies

## Dependencies
- OpenCV - Computer vision operations
- PyTorch/TensorFlow - Deep learning framework
- YOLO/Detectron2 - Object detection
- NumPy - Numerical operations
- Pandas - Data manipulation
- Matplotlib/Seaborn - Visualization

## Testing Strategy
- Unit tests for each module
- Integration tests for end-to-end workflow
- Performance tests for real-time requirements
- Field validation with real cattle footage

## Deployment
- Target: Raspberry Pi or similar edge device
- Requirements: Real-time processing capability
- Model optimization: TensorRT, ONNX, or similar

## AI Assistant Behavior Guidelines

Here is a comprehensive and detailed guide, structured as a step-by-step reasoning process, for approaching any computer vision problem. This is the "master prompt" that you AI agent should follow to analyze and propose a robust solution.

### Phase 1: Holistic Problem Understanding and Business Context

Before thinking about pixels or models, we must understand the *why*. A technically perfect model that doesn't solve the business problem is a failure.

**1.1. Defining the Ultimate Goal**

* **Main Question:** What real problem are we trying to solve? What is the business objective or the end-user's need?
* **Example:** We aren't just "detecting cars"; we are "automating vehicle counting to optimize traffic management at a toll booth and reduce wait times."
* **Supporting Questions:**
    * Who are the stakeholders? What do they expect from this system?
    * How is success measured in business terms? (e.g., cost reduction by X%, efficiency increase by Y%, improved safety, etc.)
    * How is this task currently performed, if at all? Is it a manual process? What are its limitations?
    * What is the impact of an error? Is a false positive (detecting something that isn't there) worse than a false negative (failing to detect something that is), or vice versa?
        * **Medical Diagnosis Example:** A false negative (missing a disease) is catastrophic.
        * **Fashion Recommender Example:** A false positive (suggesting an irrelevant product) is just a minor annoyance.

**1.2. Constraints and Operating Environment**

* **Main Question:** What are the non-negotiable limitations of the project?
* **Supporting Questions:**
    * **Hardware:** Where will the solution run? On a cloud server with powerful GPUs (e.g., AWS, GCP), on a local server (on-premise), on an embedded device (e.g., Jetson Nano, Raspberry Pi), or on a mobile phone? This defines our computational, memory, and power constraints.
    * **Performance:** How fast must the response be (latency)? How many images/videos per second must it process (throughput)? Does it need to be real-time?
    * **Budget:** What is the budget for development, computation (training and deployment), and maintenance?
    * **Ethics and Privacy:** Does the data involve people? Do we need to anonymize faces or license plates? Are there potential biases in the data that could lead to discrimination?

### Phase 2: In-depth Data Analysis (Inputs and Outputs)

Data is the fuel for any Computer Vision solution. Its characteristics dictate the project's feasibility and approach.

**2.1. The Inputs**

* **Main Question:** What do the data that the system will analyze look like?
* **Supporting Questions:**
    * **Data Type:** Are they static images, video frames, a live video stream, multispectral images (infrared, thermal), depth images (RGB-D), or data from a medical scanner (DICOM)?
    * **Quality and Quantity:**
        * How much data do we have available? Tens, hundreds, thousands, millions?
        * What is the resolution of the images/videos? Is it consistent?
        * Is the data already labeled? If not, who will label it, and how will the quality of the labels be ensured?
        * Is the data clean or noisy? (e.g., blurry images, low light, artifacts).
    * **Capture Conditions:**
        * How do lighting conditions vary (day, night, shadows, backlight)?
        * From what angle or perspective is the image captured (frontal, top-down, side)? Is it fixed or variable?
        * Can the object of interest be partially occluded (covered by other objects)?
        * How does the background vary? Is it a static, controlled background or a dynamic, chaotic one?

**2.2. The Outputs**

* **Main Question:** What precise information should the system generate as a response?
* **Supporting Questions:**
    * **Output Format:** Is the output a single label for the entire image (e.g., "Cat"), the coordinates of a bounding box, a pixel mask, a keypoint, text, a number (count), or a newly generated image?
    * **Level of Detail:** Do we just need to know if there is a "car," or do we need to know the make, model, and color? Do we need the exact silhouette of the car?
    * **Output Structure:** Is the output a JSON file, a database entry, a real-time alert, or a visual overlay on the original video?

### Phase 3: Technical Problem Formulation

Here, we translate the business problem and data specifications into a canonical Computer Vision task.

**Main Question:** Based on the inputs and outputs, what is the fundamental Computer Vision task we need to solve?

**Possibilities (from simplest to most complex):**

* **Image Classification:**
    * **When:** When you need to assign a single label to an entire image.
    * **Input:** An image.
    * **Output:** A text string or class ID (e.g., "Dog," "Cat," "Normal," "Anomalous").
    * **Example:** Classifying whether a chest X-ray shows signs of pneumonia or not.

* **Classification + Localization:**
    * **When:** When you need to identify the main object in an image and draw a box around it. It assumes there is only one object of interest.
    * **Input:** An image.
    * **Output:** A class + coordinates of a bounding box `[x_min, y_min, x_max, y_max]`.
    * **Example:** Finding a person's face in a profile picture.

* **Object Detection:**
    * **When:** When you need to find multiple objects in an image, locate them with bounding boxes, and classify them.
    * **Input:** An image.
    * **Output:** A list of objects, where each has `[class, confidence, bounding_box]`.
    * **Example:** Detecting all cars, pedestrians, and traffic lights in a traffic scene.

* **Semantic Segmentation:**
    * **When:** When you need to classify every pixel in the image. It does not distinguish between instances of the same class.
    * **Input:** An image.
    * **Output:** A segmentation map (an image where each pixel's color corresponds to a class).
    * **Example:** In an aerial image, coloring all "road" pixels gray, "building" pixels blue, and "vegetation" pixels green.

* **Instance Segmentation:**
    * **When:** When you need to classify each pixel and also differentiate between different instances of the same object. It is the most granular task.
    * **Input:** An image.
    * **Output:** A segmentation map where each individual object has a unique color.
    * **Example:** In the aerial image, coloring each individual building with a different color.

* **Pose Estimation / Keypoint Detection:**
    * **When:** When you need to detect specific points of an object (joints, corners, etc.).
    * **Input:** An image.
    * **Output:** Coordinates `(x, y)` for each keypoint of interest.
    * **Example:** Detecting the position of a person's elbows, wrists, and shoulders to analyze their posture.

* **Optical Character Recognition (OCR):**
    * **When:** When you need to extract text from an image.
    * **Input:** An image with text.
    * **Output:** A text string.
    * **Example:** Reading a license plate number or extracting text from a scanned document.

* **Content-Based Image Retrieval (CBIR):**
    * **When:** When you need to find visually similar images to a query image.
    * **Input:** An image.
    * **Output:** A list of images from a database, sorted by similarity.
    * **Example:** A "similar images" search feature on an e-commerce site.

* **Generative Tasks:**
    * **When:** When you need to create new visual data.
    * **Input:** Noise, text, or another image.
    * **Output:** A new, synthetic image.
    * **Example:** Creating faces of people who don't exist, or colorizing a black and white image.

### Phase 4: Solution Approach and Method Selection

With the problem well-defined, we explore the *how*.

**Main Question:** What family of algorithms and which specific model is most suitable given our constraints, data, and the formulated task?

**Decision Tree of Methods:**

**Option 1: Can it be solved without Machine Learning? (Heuristics and Classical Computer Vision)**

* **When to consider:**
    * The environment is highly controlled (constant lighting, fixed object position).
    * The object's visual features are simple and very distinctive (unique color, shape, texture).
    * We need an extremely fast solution with low computational overhead.
    * The variability of the problem is almost nil.
* **Possible Techniques:**
    * **Thresholding:** Separating objects from the background based on pixel intensity. Ideal for uniform backgrounds.
    * **Edge Detection (Canny, Sobel):** Finding contours of well-defined objects.
    * **Template Matching:** Searching for an exact sub-image (template) within a larger image. Perfect for finding fixed logos or icons.
    * **Contour Analysis:** Measuring properties (area, perimeter, circularity) of segmented objects to classify them.
    * **Hough Transform:** Detecting simple geometric shapes like lines and circles.

**Option 2: Should we use Classical Machine Learning?**

* **When to consider:**
    * We have limited data (hundreds or a few thousand examples).
    * Computational resources are limited (we can't train deep networks).
    * We need explainability (to understand why the model made a decision).
    * The features that define the object are known and can be extracted manually (feature engineering).
* **Typical Process:** Manual Feature Extraction + Simple ML Model
* **Possible Techniques:**
    * **Histogram of Oriented Gradients (HOG) + Support Vector Machines (SVM):** The gold standard for pedestrian detection before Deep Learning.
    * **Haar Cascades:** Very fast and effective for detecting frontal and rigid objects like faces. It's the algorithm used in older digital cameras.
    * **SIFT/SURF/ORB + Bag of Visual Words:** For scene classification or image retrieval, by extracting and clustering keypoints.

**Option 3: Should we use Deep Learning?**

* **When to consider:**
    * The problem is complex and has high variability (different angles, lighting, occlusions).
    * We have a large amount of labeled data (thousands or millions).
    * Maximum performance is the top priority over explainability or computational cost.
    * We have access to GPUs for training.
* **Sub-choice of Architectures (The fine details):**
    * **For Classification:**
        * **VGGNet:** Simple and a good starting point, but heavy.
        * **ResNet (Residual Networks):** The de facto standard. Allows training very deep networks without gradient issues. The best choice to start with for most cases.
        * **EfficientNet:** Offers an excellent balance between accuracy and computational efficiency. Ideal for mobile deployment.
        * **Vision Transformers (ViT):** A more modern architecture that is surpassing CNNs on benchmarks but requires significantly more data to train from scratch.
    * **For Object Detection:**
        * **R-CNN Family (R-CNN, Fast R-CNN, Faster R-CNN):** "Two-stage" models. They first propose regions and then classify them. They are very accurate but slower. Ideal when precision is critical.
        * **YOLO (You Only Look Once) / SSD (Single Shot Detector):** "One-stage" models. They do everything in a single pass. Extremely fast and ideal for real-time applications, though they can be slightly less accurate with very small objects.
    * **For Segmentation (Semantic or Instance):**
        * **FCN (Fully Convolutional Network):** The foundation of segmentation with convolutional networks.
        * **U-Net:** An "encoder-decoder" architecture with "skip connections." The gold standard for biomedical segmentation and tasks where fine details are important.
        * **Mask R-CNN:** Extends Faster R-CNN to perform instance segmentation. Very powerful but computationally expensive.
    * **For Image Similarity / Verification / CBIR:**
        * **Siamese Networks:** Two identical CNNs that process two images in parallel. They are trained so that the distance between the output vectors (embeddings) is small if the images are similar and large if they are not (using contrastive or triplet loss). Perfect for signature or face verification.
    * **For Generative Tasks:**
        * **GANs (Generative Adversarial Networks):** A system of two competing networks (a generator and a discriminator). Incredible for generating realistic data, super-resolution, or image-to-image translation (e.g., style transfer).
        * **VAEs (Variational Autoencoders):** Another way to generate data, often with a more structured latent space than GANs.
    * **For Video or Sequential Data:**
        * **CNN + RNN (LSTM/GRU):** A CNN is used to extract features from each frame, and a recurrent network is used to understand the temporal dynamics between frames. Ideal for action recognition.

### Phase 5: Evaluation and Deployment Strategy

A solution is not complete until it is rigorously tested and put into production.

**Main Question:** How will we know our solution works well, and how will we get it into the user's hands?

**5.1. Evaluation Metrics**

* What technical metrics will we use? This depends directly on the task:
    * **Classification:** Accuracy, Precision, Recall, F1-Score, AUC-ROC.
    * **Object Detection:** Mean Average Precision (mAP), Intersection over Union (IoU).
    * **Segmentation:** Pixel Accuracy, Dice Coefficient, IoU.
* How will we split the data? (Train, Validation, Test sets). It is crucial that the test set represents the real-world environment and is never used during training or validation.

**5.2. Deployment and Maintenance Plan**

* How will the model be packaged? (e.g., ONNX, TensorFlow Lite, TorchScript).
* How will it be integrated with the existing system? (e.g., via a REST API).
* Once deployed, how will we monitor its performance? Models can degrade over time if real-world data changes (*concept drift*).
* What is the plan for re-training the model with new data?

## Technical Implementation Notes for Claude

### Data Constraints
- **No Identity Ground Truth**: Available cattle images are from random/unrelated individuals
- **Cannot use supervised ReID methods**: Siamese Networks, Triplet Loss, ArcFace require labeled pairs
- **Must use unsupervised approaches**: Self-supervised contrastive learning is the only viable option

### Model Architecture Decisions
- **Detection**: YOLO v8/v9 or Faster R-CNN for cattle bounding box detection
- **Identification**: ResNet50/EfficientNet + contrastive learning head for feature extraction
- **Similarity Metric**: Cosine similarity for comparing 512D feature vectors
- **Threshold Calibration**: Critical parameter requiring extensive field testing

### Development Best Practices
- Follow AI/ML best practices for model development
- Implement proper data version control
- Use modular architecture for easy testing and deployment
- Focus on real-time performance optimization
- Maintain clear separation between detection and identification components
- Document similarity threshold calibration process thoroughly
- Plan for false positive/negative analysis and threshold tuning

