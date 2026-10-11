from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from sentiment.inference import load_model, predict_batch
from sentiment.preprocessing import clean_text


class SentimentPipelineTests(TestCase):
    def test_text_cleaning(self):
        dirty = "<p>Great product! https://example.com &amp; wonderful support  </p>"
        cleaned = clean_text(dirty)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("Great product!", cleaned)
        self.assertIn("wonderful support", cleaned)

    def test_model_loads(self):
        pipeline = load_model()
        self.assertIsNotNone(pipeline)
        self.assertTrue(hasattr(pipeline, "predict"))

    def test_batch_prediction_labels_and_confidence(self):
        texts = [
            "This is the best purchase I have ever made, love it!",
            "It functions as specified in the manual. Average performance.",
            "Horrible product, broke on day one and customer service was rude.",
            "",  # Empty test
        ]
        results = predict_batch(texts)
        self.assertEqual(len(results), 4)

        # Check positive
        self.assertEqual(results[0]["label"], "POSITIVE")
        self.assertGreater(results[0]["confidence"], 0.5)
        self.assertIn("positive_prob", results[0])

        # Check neutral
        self.assertEqual(results[1]["label"], "NEUTRAL")

        # Check negative
        self.assertEqual(results[2]["label"], "NEGATIVE")
        self.assertIsInstance(results[2]["confidence"], float)
        self.assertGreater(results[2]["confidence"], 0.35)

        # Check empty fallback
        self.assertEqual(results[3]["label"], "NEUTRAL")

    def test_model_info_api(self):
        client = APIClient()
        response = client.get("/api/model/info/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("model_version", response.data)
        self.assertIn("known_limitations", response.data)
        self.assertEqual(response.data["model_version"], "v1.0-tfidf-logreg")

    def test_model_evaluation_api(self):
        client = APIClient()
        response = client.get("/api/model/evaluation/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("evaluation_metrics", response.data)
        self.assertIn("confusion_matrix", response.data)
        metrics = response.data["evaluation_metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
