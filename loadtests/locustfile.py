"""
Customer Feedback Intelligence (CFI)
Locust Load Testing Suite for Authenticated Web and API Concurrency.
Simulates concurrent user sessions: browsing dashboard, fetching analytics,
filtering paginated reviews, querying model info, and streaming review exports.
"""

import random

from locust import HttpUser, between, task


class CFIUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Simulate user registration, CSRF handshake, and dataset seeding."""
        # 1. Fetch login page to initialize CSRF token cookie
        self.client.get("/login/", name="GET /login/ (CSRF Handshake)")
        csrf_token = self.client.cookies.get("csrftoken", "")
        auth_headers = {"X-CSRFToken": csrf_token} if csrf_token else {}

        self.username = f"locust_user_{random.randint(10000, 99999)}"
        self.password = "LoadTestPass123!"

        # 2. Register user
        reg_resp = self.client.post(
            "/api/auth/register/",
            json={
                "username": self.username,
                "email": f"{self.username}@test.com",
                "password": self.password,
                "password_confirm": self.password,
            },
            headers=auth_headers,
            name="POST /api/auth/register/",
        )
        if reg_resp.status_code != 201:
            self.client.post(
                "/api/auth/login/",
                json={"username": self.username, "password": self.password},
                headers=auth_headers,
                name="POST /api/auth/login/",
            )

        # Update csrf token after auth session establishment
        csrf_token = self.client.cookies.get("csrftoken", csrf_token)
        self.csrf_headers = {"X-CSRFToken": csrf_token} if csrf_token else {}

        # 3. Retrieve user datasets
        self.dataset_ids = []
        ds_resp = self.client.get("/api/datasets/", name="GET /api/datasets/ (Init)")
        if ds_resp.status_code == 200:
            datasets = ds_resp.json()
            self.dataset_ids = [d["id"] for d in datasets if "id" in d]

        # 4. Upload a small synthetic dataset if none exists
        if not self.dataset_ids:
            csv_content = (
                "review_id,review_text,rating,review_date,product_category\n"
                "L1,Fantastic quality and super fast shipping,5,2026-09-01,Electronics\n"
                "L2,Broken on arrival and customer support was rude,1,2026-09-02,Electronics\n"
                "L3,Average build quality works as described,3,2026-09-03,Home\n"
                "L4,Battery life lasts two full days,5,2026-09-04,Electronics\n"
                "L5,Horrible experience unit overheated,1,2026-09-05,Hardware\n"
            )
            files = {"file": ("locust_sample.csv", csv_content.encode("utf-8"), "text/csv")}
            up_resp = self.client.post(
                "/api/datasets/",
                data={"name": "Locust Ingestion Set"},
                files=files,
                headers=self.csrf_headers,
                name="POST /api/datasets/ (Upload)",
            )
            if up_resp.status_code in (201, 202):
                data = up_resp.json()
                if "dataset_id" in data:
                    self.dataset_ids.append(data["dataset_id"])

    @task(4)
    def view_dashboard(self):
        """Simulate navigating to Dashboard."""
        self.client.get("/dashboard/", name="GET /dashboard/")

    @task(5)
    def fetch_datasets_list(self):
        """Query user datasets list API."""
        self.client.get("/api/datasets/", name="GET /api/datasets/")

    @task(3)
    def view_model_info(self):
        """Inspect model architecture and evaluation metrics."""
        self.client.get("/api/model/info/", name="GET /api/model/info/")
        self.client.get("/api/model/evaluation/", name="GET /api/model/evaluation/")

    @task(6)
    def view_analytics_summary(self):
        """Fetch summary KPIs and sentiment breakdown for a dataset."""
        if self.dataset_ids:
            ds_id = random.choice(self.dataset_ids)
            self.client.get(f"/api/datasets/{ds_id}/summary/", name="GET /api/datasets/[id]/summary/")
            self.client.get(f"/api/datasets/{ds_id}/sentiment-distribution/", name="GET /api/datasets/[id]/sentiment-distribution/")
            self.client.get(f"/api/datasets/{ds_id}/trends/", name="GET /api/datasets/[id]/trends/")

    @task(5)
    def filter_reviews(self):
        """Query paginated review explorer with search and sentiment filters."""
        if self.dataset_ids:
            ds_id = random.choice(self.dataset_ids)
            sentiment = random.choice(["", "POSITIVE", "NEGATIVE", "NEUTRAL"])
            self.client.get(
                f"/api/datasets/{ds_id}/reviews/?sentiment={sentiment}&page=1",
                name="GET /api/datasets/[id]/reviews/?sentiment=[s]",
            )

    @task(2)
    def export_reviews(self):
        """Test streaming CSV export endpoint."""
        if self.dataset_ids:
            ds_id = random.choice(self.dataset_ids)
            self.client.get(f"/api/datasets/{ds_id}/reviews/export/", name="GET /api/datasets/[id]/reviews/export/")
