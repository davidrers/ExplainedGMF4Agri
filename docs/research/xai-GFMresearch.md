State-of-the-Art Research Report: Explainable AI (XAI) for Geospatial Foundation Models (GFMs)
1. Comprehensive Literature Review: Top 10 Sources
Alemohammad et al. (2025):  Introduces a global XAI benchmark for geospatial foundation models that evaluates spectral, spatial, and temporal learning properties within the embedding space using multispectral imagery.
Gavade & Gavade (2025):  Explores the integration of XAI with Transformer models and Natural Language Processing (NLP) to improve transparency and interpretability in Land Use Land Cover (LULC) classification.
Zhu et al. (2022):  Demonstrates an innovative LIME-based data selection method for GAN-generated SAR imagery, utilizing interpretability to select training samples that possess representative target features rather than spurious background speckles.
Abdollahi & Pradhan (2021):  Investigates urban vegetation mapping using SHAP to rank input parameters, proving that textural features like GLCM homogeneity are often more critical than spectral bands alone for accurate interpretation.
Ayush et al. (2020):  Develops a framework for village-level poverty estimation in Uganda by using object detection to extract human-interpretable features from high-resolution satellite imagery.
Ishikawa et al. (2023):  Proposes an example-based XAI method for remote sensing that fosters stakeholder trust by providing visual analogs to explain complex model classification decisions.
Ali et al. (2023):  Establishes a hierarchical taxonomy for XAI, categorizing techniques into data, model, and post-hoc explainability to support the development of trustworthy AI systems.
Bommer et al. (2023):  Provides a structured guide for climate science to evaluate XAI methods based on domain relevance, scalability, and the ability to maintain physical consistency.
Ibrahim & Hall (2022):  Conducts a correlation study between human expert ratings and machine estimations of poverty from satellite imagery, validating the alignment of machine features with domain knowledge.
Hall et al. (2022):  Reviews the intersection of deep learning and human development, advocating for interdisciplinary standards to ensure GFM explanations contribute to genuine scientific discovery.
2. Taxonomy of XAI Strategies in Geospatial Vision Models
Interpreting GFMs requires a selection of techniques that address the unique artifacts of remote sensing data, such as gradient noise in SAR or the long-range dependencies of Transformers.| XAI Technique | Core Mechanism | Geospatial Application || ------ | ------ | ------ || SHAP | Assigns an importance value to each feature by calculating the average marginal contribution across all possible feature subsets. | Feature ranking in LULC and quantifying the impact of indices (NDVI/NDWI) on urban vegetation mapping. || LIME | Generates a locally faithful linear surrogate model by perturbing input superpixels to approximate the complex model's decision boundary. | Detecting "Clever Hans" in SAR target recognition; ensuring models focus on scattering properties rather than background noise. || Grad-CAM | Produces localization maps by using the gradients of a target class flowing into the final convolutional layer. | Highlighting spatial regions of interest in scene classification, though often subject to upsampling artifacts. || Score-CAM | A gradient-free approach that weights feature maps by the "Increase in Confidence" (CIC) when the map is applied as a mask. | Visualizing model interest areas in hyperspectral data where gradient-based methods may be unstable. || Attention Maps | Visualizes the self-attention weights in Transformer architectures to reveal contextual dependencies between image patches. | Understanding how multi-scale spatial arrangements and temporal cues influence land cover transitions. || Rule Extraction | Simplifies complex "black-box" decisions into a set of human-readable IF-THEN logic paths or decision trees. | Validating classification logic for environmental monitoring and regulatory compliance. || Model Visualization | Reduces high-dimensional features into 2D/3D space using t-SNE or UMAP to inspect clustering behavior. | Analyzing class separation and the distribution of land cover types in the GFM embedding space. |
3. Direct vs. Downstream Application
A critical pivot in geospatial research is the shift from evaluating models via  Downstream Tasks  to evaluating the  GFM Embedding Space  directly. Traditionally, models were assessed solely on performance metrics (e.g., Top-1 accuracy) for specific tasks like LULC classification. However, as noted by the  EGU 2025 Meeting Organizer , standard benchmarks often hide the fact that models are not "physically consistent."Evaluating the embedding space immediately after pre-training—testing spectral, spatial, and temporal learning—allows researchers to quantify model characteristics before task-specific biases are introduced. This "direct" evaluation is essential for bridging the gap between Earth Observation (EO) and Climate Science, ensuring the latent representations capture fundamental Earth system properties rather than just statistical correlations.
4. The Role of Raw Satellite Imagery in Explanations
Returning to raw satellite imagery is mandatory for validating the reliability of GFMs, particularly in high-resolution or SAR contexts. The  MDPI SAR study  identifies the "Clever Hans" phenomenon, where a model achieves high accuracy by learning spurious relationships—such as the relationship between a target's class and the specific speckle noise in its background—rather than the target's physical electromagnetic scattering features.Because SAR targets are often represented by a minimal number of pixels,  superpixel segmentation  and LIME-based visualization are necessary to confirm that the positive contribution region matches the physical target. Without the raw context, it is impossible to determine if a model is ignoring the target and performing "data selection" based on background artifacts. Visual confirmation against raw imagery ensures the model has learned the intended physical scattering properties.
5. Testing and Benchmarking XAI Approaches
To move beyond qualitative "eye-tests," GFMs require rigorous methodologies to validate explanation reliability:
Perturbation & Masking:  Sensitivity is tested by systematically turning superpixels "on or off." If masking a region identified by LIME as a high contributor does not significantly change the model output, the explanation is deemed unfaithful.
Synthetic Benchmarks:  Models are challenged with image chips featuring  homogeneous spatial patterns  (to isolate spectral tasks) and  heterogeneous features  (to represent the true spatial distribution of land cover).
Embedding Space Analysis:  Testing specific properties in the latent space validates whether the GFM internalized multispectral and temporal features during self-supervised pre-training.
Temporal Disturbances:  The use of time-series chips showing  pre- and post-event  imagery (e.g., wildfire or flood disturbances) validates the model's ability to interpret the "time variable."
Quantitative Metrics:  Reliability is measured using  Mutual Information (MI)  to ensure the independence of generated features and the  GAN-test , which uses a classifier trained on real data to validate the authenticity and distribution mapping of generated imagery.
6. Critical Factors for Geospatial Interpretation
To be considered "physically consistent" and scientifically sound, GFMs must account for factors grounded in Earth system physics, such as the Navier-Stokes equations for fluid dynamics or spectral reflectance properties:
Spectral Band Attribution:  Models must demonstrate an understanding of band-specific contributions, specifically identifying how indices like  NDVI  (vegetation) and  NDWI  (water) drive classification.
Temporal Dynamics:  Interpretation must include the time variable to recognize disturbances and seasonal phenology, ensuring the model distinguishes between permanent land cover changes and transient events.
Spatial Texture & arrangement:  Using Gray-Level Co-occurrence Matrix ( GLCM ) features, models must explain decisions based on  Homogeneity  and  Dissimilarity . This is vital for differentiating classes that are spectrally similar but structurally distinct, such as different types of urban vegetation.
7. Challenges and The Way Forward
The "black-box" nature of large Transformers remains a significant barrier. The way forward requires models that move beyond statistical mapping to become "ideal Earth FMs" that are task-agnostic and multisensory.Future Research Checklist:
  Wavelength Embedding:  Directly incorporating spectral wavelength information into the model architecture.
  Geolocation Embedding:  Integrating precise spatial coordinates to ensure regional awareness.
  Scale Awareness:  Maintaining interpretability across varying spatial resolutions (from 0.1m to 30m).
  Multisensory Integration:  Combining Optical, SAR, Radar, and Meteorological data in a unified framework.
  Task-Agnostic Pre-training:  Developing representations that serve general purposes without being over-fit to specific labels.
  Uncertainty Quantification:  Providing confidence intervals to prevent overconfidence in high-stakes climate decisions.
  Physical Consistency:  Ensuring model outputs align with known physical laws (e.g., mass and energy balance).
  Carbon Minimization:  Optimizing training and adaptation (e.g., Parameter Efficient Fine Tuning) to reduce the environmental footprint of AI research.


  Briefing Document: Explainable AI (XAI) and Geospatial Foundation Models
Executive Summary
The rapid advancement of Artificial Intelligence (AI) in geospatial and climate sciences is currently transitioning from task-specific models to broad, self-supervised  Foundation Models (FMs) . While these models offer unprecedented potential for Earth observation (EO) and climate modeling, they often operate as "black boxes," raising significant concerns regarding transparency, reliability, and the  "Clever Hans" phenomenon —where models achieve high accuracy by learning spurious correlations rather than true causal features.Explainable AI (XAI)  has emerged as a critical solution to bridge this gap, providing the interpretability necessary for high-stakes decisions in agriculture, land management, and disaster response. Recent research highlights a strategic shift toward:
Comprehensive Benchmarking:  Developing standardized tests to evaluate whether models truly learn spectral, spatial, and temporal properties.
Targeted Data Selection:  Using XAI techniques like  LIME  and  SHAP  to refine training datasets and ensure models focus on meaningful physical targets rather than background noise.
Integrative Architectures:  Combining FMs with  Transformers  and  Natural Language Processing (NLP)  to extract insights from multi-modal data sources.The ultimate goal is the development of an  Ideal Earth FM  that is multisensory, scale-aware, geolocated, and physically consistent, while maintaining a minimized carbon footprint.
1. The Paradigm Shift to Earth Foundation Models (FMs)
1.1 Defining the Foundation Model
A Foundation Model is characterized by two primary traits:
Broad Training:  It is trained on vast, generally unlabeled datasets using self-supervision.
Adaptability:  It can be fine-tuned or adapted to a wide range of downstream tasks (e.g., object recognition, information extraction, instruction following).
1.2 Potential and Necessity
The adoption of FMs in Earth and climate science aims to:
Unlock Big Data:  Effectively utilize the petabytes of open data from Landsat, Sentinel, MODIS, and ERA5.
Enhance Label Efficiency:  Reduce the reliance on expensive, manually labeled datasets.
Improve Modeling:  Bridge the gap between EO and climate science to create more accurate Earth system models.
Environmental Efficiency:  Reduce carbon footprints through parameter-efficient fine-tuning rather than training models from scratch for every task.
1.3 Features of an Ideal Earth FM
Current research identifies "must-have" and "highly desirable" features for the next generation of models:| Category | Essential Features || ------ | ------ || Operational | Geolocation embedding, scale awareness, wavelength embedding, and time variables. || Structural | Multisensory capabilities and task-agnostic design. || Scientific | Physical consistency (adherence to laws of physics) and uncertainty quantification. || Ethical | Balanced geographical representation and minimized carbon footprint. |
2. Explainable AI (XAI) in Geospatial Analysis
2.1 The Transparency Crisis: The "Clever Hans" Phenomenon
A major risk in Deep Neural Networks (DNNs) is the "Clever Hans" effect, where a model provides correct predictions based on the "wrong" criteria. In SAR (Synthetic Aperture Radar) image recognition, a model might categorize a target correctly not by identifying the vehicle, but by identifying scattered speckles in the background unique to that training set.
2.2 Core XAI Techniques
XAI aims to decompose the intricacies of complex models to provide human-understandable explanations:
Local Interpretable Model-Agnostic Explanations (LIME):  Approximates a complex model locally with a simpler, interpretable one. It is superior to some methods because it identifies both  positive and negative contributions  to a decision.
SHapley Additive exPlanations (SHAP):  Quantifies the contribution of individual features to a prediction, often used in Land Use Land Cover (LULC) classification and poverty mapping.
Class Activation Mapping (CAM):  Visualizes areas of interest by highlighting regions that contribute positively to a classification, though it often suffers from lack of precision compared to LIME.
2.3 XAI for Data Selection and Quality
XAI is not just for post-hoc analysis but can be used to improve model training:
LIME-based Selection:  By visualizing positive contributions, researchers can discard training images where the model focuses on the background.
Improved GANs:  Using XAI-selected data for Generative Adversarial Networks (GANs) results in generated images with clearer contours, finer textures, and higher authenticity (GAN-test scores).
3. Transforming Land Use Land Cover (LULC) Classification
LULC classification is pivotal for sustainable land management, irrigation, and biodiversity conservation. Traditionally, these models were subjective and resource-intensive.
3.1 Technological Integration
The integration of XAI with advanced architectures is refining LULC categorizations:
Transformer Models:  Capture complex contextual cues and spatial relationships within satellite imagery.
Natural Language Processing (NLP):  Encodes geospatial vector data and extracts insights from textual metadata and reports.
Multi-source Data:  Combining satellite imagery, aerial photographs, and textual data to capture a comprehensive understanding of land dynamics.
3.2 Benefits for Stakeholders
Informed Decision-Making:  Provides actionable insights from complicated geographic data.
Trust and Accountability:  Demonstrates the reasoning behind classifications, essential for regulatory compliance.
Error Detection:  Facilitates the identification of biases or incorrect categorizations at a granular (pixel or segment) level.
4. Benchmarking and Evaluation Challenges
4.1 A New XAI Benchmark for Geospatial FMs
Recent initiatives have introduced global benchmarks to quantify foundation model characteristics post-pre-training. This involves testing the model's embedding space across three specific tasks:
Spectral Task:  Homogeneous spatial patterns from major land cover classes.
Spatial Task:  Heterogeneous features representative of true distribution.
Temporal Task:  Time-series imagery of disturbances, such as wildfires and floods.
4.2 Gaps in Current Benchmarking
Despite progress, several deficiencies remain in evaluating Earth FMs:
Lack of Standardization:  No consensus on FM adaptation protocols.
Geographic Bias:  Certain global regions remain underrepresented.
Real-world Applicability:  A lack of scenarios reflecting complex, multi-variable real-world conditions.
Disconnected Domains:  Insufficient connection between EO-specific benchmarks and climate-specific benchmarks.
5. The Way Forward: Future Research Directions
To fully exploit the potential of AI in Earth sciences, the document context suggests several priority areas:
Comprehensive Metadata Integration:  Leveraging dynamic encoders and spatial-temporal modeling to better handle metadata.
Geographical Mixture of Experts:  Developing models that can adapt their expertise based on the specific geographic region being analyzed.
Continual Learning:  Establishing frameworks for "Machine Unlearning" (privacy) and "Continual Pre-training" to keep models updated without full retraining.
Adversarial Defenses:  Strengthening models against intentional attacks or data corruption.
Physical Consistency:  Moving beyond pattern recognition to ensure AI predictions do not violate fundamental physical constants or laws.



