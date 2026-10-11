"""
Customer Feedback Intelligence (CFI)
Performance and Throughput Benchmarking Suite.
Measures CSV ingestion, batch sentiment inference, database bulk write, and API query latency.
"""

import os
import random
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Setup Django environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django

django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402

from analytics.services import get_summary_metrics  # noqa: E402
from datasets.models import Dataset, DatasetStatus  # noqa: E402
from datasets.services import run_dataset_ingestion  # noqa: E402
from reviews.models import Review  # noqa: E402
from sentiment.inference import predict_batch  # noqa: E402

User = get_user_model()

SAMPLE_PHRASES = [
    ("Great product, really loved the build quality and performance!", 5, "POSITIVE", "Electronics"),
    ("Customer support resolved my issue quickly and professionally.", 5, "POSITIVE", "Services"),
    ("Battery life is impressive, easily lasts a full day of heavy use.", 4, "POSITIVE", "Electronics"),
    ("Fast shipping, arrived in perfect condition right on schedule.", 5, "POSITIVE", "Logistics"),
    ("Acceptable item for the price point, but build is basic.", 3, "NEUTRAL", "Home"),
    ("It works as specified in the manual. Standard performance.", 3, "NEUTRAL", "Electronics"),
    ("Delivery took exactly the estimated time, nothing special.", 3, "NEUTRAL", "Logistics"),
    ("Average experience overall. Does what it needs to do.", 3, "NEUTRAL", "Home"),
    ("Terrible quality, broke after two days of use. Avoid this!", 1, "NEGATIVE", "Electronics"),
    ("Customer service was completely unhelpful and rude.", 1, "NEGATIVE", "Services"),
    ("Package arrived damaged and two weeks late with no tracking.", 1, "NEGATIVE", "Logistics"),
    ("Horrible software update broke all previous settings.", 2, "NEGATIVE", "Software"),
]


def generate_synthetic_csv(num_rows: int) -> str:
    """Generate a temporary CSV file with synthetic reviews."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8")
    temp_file.write("review_id,review_text,rating,review_date,product_category\n")

    base_date = date(2026, 1, 1)

    for i in range(1, num_rows + 1):
        phrase, rating, _, category = random.choice(SAMPLE_PHRASES)
        r_date = base_date + timedelta(days=random.randint(0, 180))
        # Add variation
        text = f"{phrase} (Transaction #{i})"
        temp_file.write(f"BENCH-{i:06d},\"{text}\",{rating},{r_date.isoformat()},{category}\n")

    temp_file.close()
    return temp_file.name


def run_benchmark(dataset_sizes=(1000, 5000)):
    """Run full benchmark across specified dataset sizes."""
    print("=" * 70)
    print("CUSTOMER FEEDBACK INTELLIGENCE (CFI) BENCHMARK SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
    print("=" * 70)

    # Ensure benchmark user exists
    user, _ = User.objects.get_or_create(username="benchmark_user", defaults={"email": "bench@cfi.test"})

    results = []

    for size in dataset_sizes:
        print(f"\n[+] Running Benchmark for {size:,} Reviews...")

        # 1. Generate CSV
        gen_start = time.perf_counter()
        csv_path = generate_synthetic_csv(size)
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        gen_duration = time.perf_counter() - gen_start
        print(f"    - Generated synthetic CSV ({file_size_mb:.2f} MB) in {gen_duration:.2f}s")

        # 2. Measure Ingestion + Batched Inference
        with open(csv_path, "rb") as f:
            uploaded_file = SimpleUploadedFile(f"bench_{size}.csv", f.read(), content_type="text/csv")

        dataset = Dataset.objects.create(
            owner=user,
            name=f"Benchmark Dataset {size}",
            original_filename=f"bench_{size}.csv",
            file=uploaded_file,
            file_size_bytes=uploaded_file.size,
            status=DatasetStatus.QUEUED,
        )

        ingest_start = time.perf_counter()
        success = run_dataset_ingestion(str(dataset.id))
        ingest_duration = time.perf_counter() - ingest_start

        if not success:
            print("    [!] Ingestion failed!")
            continue

        throughput = size / ingest_duration if ingest_duration > 0 else 0
        print(f"    - Ingestion & Sentiment Classification Time: {ingest_duration:.2f}s")
        print(f"    - Ingestion Throughput: {throughput:.1f} reviews/sec")

        # 3. Measure Isolated Inference Throughput (1,000 texts batch)
        sample_texts = [random.choice(SAMPLE_PHRASES)[0] for _ in range(1000)]
        inf_start = time.perf_counter()
        predict_batch(sample_texts)
        inf_duration = time.perf_counter() - inf_start
        inf_throughput = 1000 / inf_duration if inf_duration > 0 else 0
        print(f"    - Isolated Model Inference Speed: {inf_throughput:.1f} predictions/sec ({inf_duration*1000:.1f} ms for 1,000 batch)")

        # 4. Measure Summary SQL Aggregation Latency
        agg_latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            get_summary_metrics(dataset, use_cache=False)
            agg_latencies.append((time.perf_counter() - t0) * 1000)

        p50_agg = sorted(agg_latencies)[len(agg_latencies) // 2]
        print(f"    - Database Summary Aggregation Latency (uncached p50): {p50_agg:.2f} ms")

        # 5. Measure Paginated Reviews Query Latency (page_size=20)
        review_latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            list(Review.objects.filter(dataset=dataset).select_related("sentiment")[:20])
            review_latencies.append((time.perf_counter() - t0) * 1000)

        p50_rev = sorted(review_latencies)[len(review_latencies) // 2]
        print(f"    - Paginated Review Query (select_related p50): {p50_rev:.2f} ms")

        # Cleanup
        try:
            os.remove(csv_path)
            dataset.delete()
        except Exception:
            pass

        results.append({
            "size": size,
            "duration_s": round(ingest_duration, 2),
            "throughput_rps": round(throughput, 1),
            "inference_rps": round(inf_throughput, 1),
            "agg_latency_ms": round(p50_agg, 2),
            "review_latency_ms": round(p50_rev, 2),
        })

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 70)
    print(f"{'Reviews':<10} | {'Total Time (s)':<14} | {'Ingest (rec/s)':<16} | {'Inference (rec/s)':<18} | {'Agg p50 (ms)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['size']:<10} | {r['duration_s']:<14} | {r['throughput_rps']:<16} | {r['inference_rps']:<18} | {r['agg_latency_ms']:<12}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark(dataset_sizes=(1000, 3000))
