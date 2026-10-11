"""
Customer Feedback Intelligence (CFI)
Locust Load Testing Suite for Authenticated Web and API Concurrency.
Simulates concurrent user sessions: browsing dashboard, fetching analytics, filtering reviews, and querying model info.
"""

import random

from locust import HttpUser, between, task


class CFIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Simulate user login and obtain session cookie."""
        self.username = f"locust_user_{random.randint(1000, 9999)}"
        self.password = "LoadTestPass123!"

        # Register user
        reg_resp = self.client.post(
            "/api/auth/register/",
            json={
                "username": self.username,
                "email": f"{self.username}@test.com",
                "password": self.password,
                "password_confirm": self.password,
            },
        )
        if reg_resp.status_code != 201:
            # Login if already exists
            self.client.post(
                "/api/auth/login/",
                json={"username": self.username, "password": self.password},
            )

        # Get list of datasets to interact with
        ds_resp = self.client.get("/api/datasets/")
        self.dataset_ids = []
        if ds_resp.status_code == 200:
            datasets = ds_resp.json()
            self.dataset_ids = [d["id"] for d in datasets]

    @task(3)
    def view_dashboard(self):
        """Simulate navigating to Dashboard."""
        self.client.get("/dashboard/", name="/dashboard/")

    @task(4)
    def fetch_datasets_list(self):
        """Query user datasets list API."""
        self.client.get("/api/datasets/", name="/api/datasets/")

    @task(5)
    def view_model_info(self):
        """Inspect model architecture and evaluation metrics."""
        self.client.get("/api/model/info/", name="/api/model/info/")
        self.client.get("/api/model/evaluation/", name="/api/model/evaluation/")

    @task(4)
    def view_analytics_summary(self):
        """Fetch summary KPIs for a dataset."""
        if self.dataset_ids:
            ds_id = random.choice(self.dataset_ids)
            self.client.get(f"/api/datasets/{ds_id}/summary/", name="/api/datasets/[id]/summary/")
            self.client.get(f"/api/datasets/{ds_id}/sentiment-distribution/", name="/api/datasets/[id]/sentiment-distribution/")

    @task(3)
    def filter_reviews(self):
        """Query paginated review explorer with search and sentiment filters."""
        if self.dataset_ids:
            ds_id = random.choice(self.dataset_ids)
            sentiment = random.choice(["", "POSITIVE", "NEGATIVE", "NEUTRAL"])
            page = random.randint(1, 3)
            self.client.get(
                f"/api/datasets/{ds_id}/reviews/?sentiment={sentiment}&page={page}",
                name="/api/datasets/[id]/reviews/?sentiment=[s]&page=[p]",
            )
