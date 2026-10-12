"""
Customer Feedback Intelligence (CFI)
Performance and Throughput Benchmarking Suite.
Measures CSV ingestion, batch sentiment inference, database bulk write, memory usage,
database query counts, and API query latency across 1,000, 10,000, and 50,000 row datasets.
"""

import argparse
import os
import random
import sys
import tempfile
import time
import tracemalloc
from datetime import date, datetime, timedelta
from pathlib import Path

# Setup Django environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402
from django.contrib.auth import get_user_model  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.db import connection, reset_queries  # noqa: E402

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
        text = f"{phrase} (Transaction #{i})"
        temp_file.write(f"BENCH-{i:06d},\"{text}\",{rating},{r_date.isoformat()},{category}\n")

    temp_file.close()
    return temp_file.name


def run_benchmark(dataset_sizes=(1000, 10000, 50000)):
    """Run full benchmark across specified dataset sizes."""
    print("=" * 80)
    print("CUSTOMER FEEDBACK INTELLIGENCE (CFI) BENCHMARK SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
    print(f"Active Database Engine: {settings.DATABASES['default']['ENGINE']}")
    print("=" * 80)

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

        # 2. Measure Ingestion + Batched Inference with Tracemalloc & Query Tracking
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

        # Track queries and peak memory
        reset_queries()
        tracemalloc.start()
        ingest_start = time.perf_counter()

        success = run_dataset_ingestion(str(dataset.id))

        ingest_duration = time.perf_counter() - ingest_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        query_count = len(connection.queries)
        peak_mb = peak_bytes / (1024 * 1024)

        if not success:
            print("    [!] Ingestion failed!")
            continue

        throughput = size / ingest_duration if ingest_duration > 0 else 0
        print(f"    - Ingestion & Sentiment Classification Time: {ingest_duration:.2f}s")
        print(f"    - Ingestion Throughput: {throughput:.1f} reviews/sec")
        print(f"    - Peak Ingestion Memory: {peak_mb:.2f} MB")
        print(f"    - Database Queries Executed: {query_count:,} queries (avg {size/max(query_count, 1):.0f} rows/query)")

        # 3. Measure Isolated Inference Throughput (1,000 texts batch)
        sample_texts = [random.choice(SAMPLE_PHRASES)[0] for _ in range(1000)]
        inf_start = time.perf_counter()
        predict_batch(sample_texts)
        inf_duration = time.perf_counter() - inf_start
        inf_throughput = 1000 / inf_duration if inf_duration > 0 else 0
        print(f"    - Isolated Model Inference Speed: {inf_throughput:.1f} predictions/sec ({inf_duration*1000:.1f} ms for 1,000 batch)")

        # 4. Measure Summary SQL Aggregation Latency (p50 / p95)
        agg_latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            get_summary_metrics(dataset, use_cache=False)
            agg_latencies.append((time.perf_counter() - t0) * 1000)

        agg_sorted = sorted(agg_latencies)
        p50_agg = agg_sorted[int(len(agg_sorted) * 0.50)]
        p95_agg = agg_sorted[int(len(agg_sorted) * 0.95)]
        print(f"    - Database Summary Aggregation Latency (uncached): p50={p50_agg:.2f} ms | p95={p95_agg:.2f} ms")

        # 5. Measure Paginated Reviews Query Latency (page_size=20, p50 / p95)
        review_latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            list(Review.objects.filter(dataset=dataset).select_related("sentiment")[:20])
            review_latencies.append((time.perf_counter() - t0) * 1000)

        rev_sorted = sorted(review_latencies)
        p50_rev = rev_sorted[int(len(rev_sorted) * 0.50)]
        p95_rev = rev_sorted[int(len(rev_sorted) * 0.95)]
        print(f"    - Paginated Review Query (select_related): p50={p50_rev:.2f} ms | p95={p95_rev:.2f} ms")

        # Cleanup dataset from database to free disk space
        try:
            os.remove(csv_path)
            dataset.delete()
        except Exception:
            pass

        results.append({
            "size": size,
            "duration_s": round(ingest_duration, 2),
            "throughput_rps": round(throughput, 1),
            "peak_mb": round(peak_mb, 2),
            "query_count": query_count,
            "inference_rps": round(inf_throughput, 1),
            "agg_p50_ms": round(p50_agg, 2),
            "agg_p95_ms": round(p95_agg, 2),
            "rev_p50_ms": round(p50_rev, 2),
            "rev_p95_ms": round(p95_rev, 2),
        })

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)
    headers = f"{'Rows':<8} | {'Time (s)':<9} | {'Throughput':<12} | {'Peak Mem':<9} | {'DB Queries':<11} | {'Inf (rps)':<10} | {'Agg p50/p95 (ms)':<16} | {'Rev p50/p95 (ms)'}"
    print(headers)
    print("-" * 80)
    for r in results:
        agg_str = f"{r['agg_p50_ms']:.2f} / {r['agg_p95_ms']:.2f}"
        rev_str = f"{r['rev_p50_ms']:.2f} / {r['rev_p95_ms']:.2f}"
        print(f"{r['size']:<8} | {r['duration_s']:<9.2f} | {r['throughput_rps']:<12.1f} | {r['peak_mb']:<9.2f} | {r['query_count']:<11} | {r['inference_rps']:<10.1f} | {agg_str:<16} | {rev_str}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFI Benchmark Runner")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 10000, 50000],
        help="Dataset row sizes to benchmark (e.g. --sizes 1000 10000 50000)",
    )
    args = parser.parse_args()
    run_benchmark(dataset_sizes=tuple(args.sizes))
