# Age Prediction Models Comparison

This document provides a performance benchmark comparing three different models used for age prediction:

1. **MAE Model**: An Age Regression Model based on the MAE architecture, trained on the **UTKFace** dataset.
2. **FairFace Model**: An Age Regression Model trained on the **FairFace** dataset.
3. **Gemini Model**: Google's Gemini multimodal LLM, accessed via API for zero-shot age estimation.

---

## Benchmark Results

The benchmark was run on a test dataset of **26 labeled images**. The evaluation metrics used are:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **Average Inference Time per Image**

| Model                    | Count | MAE (Years) | RMSE (Years) | Avg Inference Time (ms) |
| ------------------------ | ----: | ----------: | -----------: | ----------------------: |
| **Gemini (Sync)**        |    26 |       3.769 |        4.828 |                  4827.0 |
| **Gemini (Async Batch)** |   14* |   **4.429** |    **5.695** |                   774.1 |
| **MAE (UTK)**            |    26 |       6.767 |        8.279 |                **54.8** |
| **FairFace**             |    26 |       6.828 |        8.713 |                    78.0 |

> **Note:** Lower values are better for MAE and RMSE.
> Inference time was measured on CPU.

---

## Key Observations

### 1. Accuracy vs. Speed Trade-off

* **Gemini** is significantly more accurate than the local models, achieving a Mean Absolute Error of approximately **3.8 years**.
* By switching to **asynchronous batch processing**, the effective throughput improved from roughly **4.8 seconds per image** to around **774 milliseconds per image**.
* Because all 26 requests were sent simultaneously using the free-tier Gemini API, some requests failed due to API rate limiting (`429 Too Many Requests`). This reduced the evaluated sample count for the async benchmark.
* In a production system, this issue could be mitigated using:

  * Retry mechanisms
  * Request throttling
  * Paid API tiers with higher limits

Meanwhile:

* **MAE** and **FairFace** provide extremely fast local inference (~55–80 ms/image on CPU).
* These models are highly suitable for:

  * Real-time applications
  * Video stream processing
  * Edge deployment
  * Offline inference

---

## 2. Local Models Comparison

* **MAE (UTK)** slightly outperformed **FairFace** on this benchmark:

  * Better MAE
  * Better RMSE
  * Faster inference speed
* The close performance between the two models suggests that both the **UTKFace** and **FairFace** datasets provide strong foundations for age regression tasks.
* However, the MAE-based model demonstrated a small but consistent advantage in this evaluation.

---

## 3. Deployment Considerations

### When to Use Gemini

Use Gemini when:

* Maximum accuracy is required
* Processing can be done offline or asynchronously
* Latency is not critical
* Internet/API access is available

### When to Use MAE / FairFace

Use MAE or FairFace when:

* Real-time inference is required
* Running on edge devices or CPUs
* Offline capability is important
* Low latency and high throughput are priorities
* API costs and dependencies should be avoided

---

## Conclusion

While Gemini delivers state-of-the-art accuracy for age estimation, dedicated regression models such as **MAE** and **FairFace** remain the more practical option for production environments that demand high throughput, low latency, and offline capability.

Among the local models evaluated, the **MAE model trained on UTKFace** achieved the best balance of:

* Accuracy
* Inference speed
* Deployment efficiency

making it the strongest local candidate for real-time age prediction systems.
