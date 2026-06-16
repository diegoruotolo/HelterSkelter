# Helter Skelter

## Overview

Helter Skelter is a **native Scala and Apache Spark implementation** of the Meta (Facebook) Prophet algorithm for:

- 📈 Time-series forecasting
- 🚨 Anomaly detection
- 📊 Multi-metric modeling at scale

Like Prophet, it models complex signals with **four concurrent seasonalities**:

- Hourly
- Weekly
- Monthly
- Yearly

---

## Key Innovation

Helter Skelter introduces a **native multi-metric architecture**:

- A **single shared feature vector** is computed once
- Multiple metrics are modeled in sequence
- Each metric has its own regression coefficients

👉 This enables **single-pass evaluation** of multiple KPIs (e.g. bookings + searches together).

---

## Why Helter Skelter?

Standard Prophet:

- ❌ R and Python
- ❌ Requires `pandas_udf` (heavy serialization overhead)
- ❌ Single-node execution
- ❌ One model per metric

Helter Skelter:

- ✅ Fully JVM-native (Scala/Spark)
- ✅ Distributed execution across Spark executors
- ✅ No Python dependency
- ✅ Multi-metric in one pass
- ✅ Real-time scoring compatible

---

## Quick Start

### Installation

**sbt**
```scala
libraryDependencies += "io.github.diegoruotolo" %% "helter-skelter" % "0.0.1"
```

**Maven**
```xml
<dependency>
    <groupId>io.github.diegoruotolo</groupId>
    <artifactId>helter-skelter_2.12</artifactId>
    <version>0.0.1</version>
</dependency>
```

> For Scala 2.13 replace the artifactId suffix with `helter-skelter_2.13`.
> Spark (`spark-core`, `spark-sql`, `spark-mllib`) is a `provided` dependency — it must be on the classpath of your application or cluster.

---

### Complete End-to-End Example

```scala
import org.apache.spark.sql.SparkSession
import com.amadeus.helterskelter._

object HelterSkelterExample {

  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession =
      SparkSession.builder()
        .appName("HelterSkelterExample")
        .master("local[*]")
        .getOrCreate()

    // Multiple metrics can be modeled together by passing a set of column names
    val monitoredMetrics = Set("bookings", "searches")

    // ======================================================
    // 1. TRAINING PHASE (Batch job - typically scheduled)
    // ======================================================
    
    // Read the historical data for training (must include the monitored metrics and a timestamp column)
    val historicalDataDf = spark.read.parquet("hdfs:///analytics/history")

    // Configure the model (e.g. change the number of changepoints)
    val config = HSConfig(
      nChangepoints = 10
    )

    val model =
      HelterSkelter.fit(
        historicDf = historicalDataDf,
        valueCols  = monitoredMetrics,
        config     = config
      )

    // ======================================================
    // 2. SCORING PHASE (Batch or Streaming)
    // ======================================================
    val liveStreamDf =
      spark.readStream.parquet("hdfs:///analytics/live")

   
    val scoredMetricsDf =
      HelterSkelter
        .predict(model = model,
          df = liveStreamDf,
          valueCols = monitoredMetrics,
          sigmaThreshold = 3.0
        )

    // Example sink (console)
    scoredMetricsDf.writeStream
      .format("console")
      .start()
      .awaitTermination()
  }
}
```

## Output Columns

For each metric `X`, the scoring phase appends the following columns:

| Column | Type | Description |
|--------|------|------------|
| `X_forecast` | Double | Predicted baseline value |
| `X_residual` | Double | Difference: observed − forecast |
| `X_z_score` | Double | Residual normalized by training standard deviation |
| `X_pct_deviation` | Double | Percentage deviation vs forecast (null if forecast ≈ 0) |
| `X_is_anomaly` | Boolean | True if an anomaly is detected (based on threshold) |
| `X_severity` | String | Severity level: `LOW`, `MEDIUM`, `HIGH`, `OK` |
| `X_minForecast` | Double | Lower bound of expected range |
| `X_maxForecast` | Double | Upper bound of expected range |

---

## Differences with Prophet

| Aspect | Facebook Prophet                     | Helter Skelter           |
|--------|--------------------------------------|--------------------------|
| Implementation | R o Python                           | Scala (JVM-native)       |
| Execution | Single-node                          | Distributed Spark        |
| Multi-metric support | One model per metric                 | Multi-metric in one pass |
| Monthly seasonality | ❌ Must be added manually             | ✅ Built-in               |
| Multiplicative seasonality | ✅ Supported                          | ❌ Not supported          |
| Public holidays | ✅ Built-in for many countries        | ❌ Not supported  |
| Confidence band | Grows wider the further you forecast | Fixed width forever |
| Regression | L1 (Lasso)                           | L2 (Ridge) |
| Python dependency | Required                             | None |

**Bottom line**: Prophet is more statistically sophisticated and honest about what it doesn't know. Helter Skelter trades some of that sophistication for the ability to run at scale in a production streaming pipeline on the JVM, with no Python dependency.

---

## References

- 📐 [Mathematical Foundation](./docs/MathematicalFoundation.md)
- 📖 [Complete API Reference](./docs/Technical.md)
- 🎸 [Helter Skelter](https://en.wikipedia.org/wiki/Helter_Skelter_(song)) — The Beatles song that inspired the name
