# Restaurant Review Analyzer

## 1. Problem Definition
Classify restaurant feedback into sentiment categories (Positive, Neutral, Negative) and map reviews to operational topics (e.g., Food Quality, Service, Waiting Time) using NLP models to help managers address customer concerns.

## 2. Data Collection
Loaded review texts and candidate topic labels (`Food Quality`, `Service`, `Price`, `Cleanliness`, `Atmosphere`, `Location`, `Waiting Time`) directly within the Python script workflow.

## 3. Data Preparation
Formatted review strings and converted zero-shot topic confidence output scores into structured Pandas DataFrames for visualization and reporting.

## 4. Model Selection
Selected three sentiment classification models and one zero-shot topic classifier:
* `cardiffnlp/twitter-roberta-base-sentiment-latest`
* `cardiffnlp/twitter-xlm-roberta-base-sentiment`
* `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
* Zero-Shot Classifier: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

## 5. Model Card Investigation
Verified model tasks, languages, and pipeline integration requirements for `text-classification` and `zero-shot-classification` pipelines using Hugging Face Transformers.

## 6. Initial Inference Results
Executed initial inference passes using standard pipeline calls (`sentiment_pipe(user_input)` and `zeroshot_pipe(user_input, candidate_labels)`), extracting label names and confidence scores.

## 7. Evaluation
Measured prediction confidence scores and topic probability distributions across candidate labels for input reviews.

## 8. Error Analysis
Handled missing/empty string inputs by returning UI warnings and managed tokenizer compatibility exceptions across different multilingual model backends.

## 9. Model Comparison
Allowed dynamic user selection across the three sentiment models within the Streamlit sidebar to compare predicted labels and confidence outputs on identical input texts.

## 10. Limitations
Inference is constrained to single text inputs per run, depends on active model downloads from Hugging Face Hub, and relies on pre-trained zero-shot topic definitions without custom domain fine-tuning.

**Live Demo:** [https://restaurantreviewanalyzer.streamlit.app/](https://restaurantreviewanalyzer.streamlit.app/)
