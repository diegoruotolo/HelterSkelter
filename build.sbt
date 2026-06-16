import sbt._
import Keys._
import sbtrelease.ReleaseStateTransformations._
import sbtrelease.ReleasePlugin.autoImport._

lazy val root = (project in file("."))
  .settings(
    name         := "helter-skelter",
    organization := "io.github.diegoruotolo",

    // ── Maven Central metadata ──────────────────────────────────────────────
    description := "Native Scala/Spark implementation of the Prophet algorithm for time-series forecasting and anomaly detection",
    licenses    := Seq("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0")),
    homepage    := Some(url("https://github.com/diegoruotolo/HelterSkelter")),
    scmInfo := Some(ScmInfo(
      url("https://github.com/diegoruotolo/HelterSkelter"),
      "scm:git@github.com:diegoruotolo/HelterSkelter.git"
    )),
    developers := List(
      Developer(
        id    = "diegoruotolo",
        name  = "Diego Ruotolo",
        email = "",
        url   = url("https://github.com/diegoruotolo")
      )
    ),

    // ── Sonatype / Maven Central publishing ────────────────��───────────────
    sonatypeCredentialHost := "central.sonatype.com",
    sonatypeRepository     := "https://central.sonatype.com/api/v1/publisher",
    publishTo              := sonatypePublishToBundle.value,
    publishMavenStyle      := true,
    sonatypeTimeoutMillis := 600000,  // 10 minutes (default is ~60s)

    // Do not publish test artifacts
    Test / publishArtifact := false,
    // Publish sources and Scaladoc jars (required by Maven Central)
    Compile / packageSrc / publishArtifact := true,
    Compile / packageDoc / publishArtifact := true,

    // ── sbt-release configuration ──────────────────────────────────────────
    // Cross-publish both Scala versions on release
    releaseCrossBuild := true,
    releaseProcess := Seq[ReleaseStep](
      checkSnapshotDependencies,
      inquireVersions,
      runClean,
      runTest,
      setReleaseVersion,
      commitReleaseVersion,
      tagRelease,
//      releaseStepCommand("sonatypeBundleClean"),
//      releaseStepCommandAndRemaining("+publishSigned"),
//      releaseStepCommand("sonatypeBundleRelease"),
      setNextVersion,
      commitNextVersion,
      pushChanges
    ),

    // 1. Define the primary Scala version and the supported cross-build versions
    scalaVersion := "2.12.19",
    crossScalaVersions := Seq("2.12.19", "2.13.12"),

    // 2. Dynamic Java target compilation based on the active Scala version
    javacOptions ++= {
      CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, 13)) => Seq("-source", "17", "-target", "17") // Java 17 for Scala 2.13
        case _             => Seq("-source", "8", "-target", "8") // Java 8 fallback for Scala 2.12
      }
    },

    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Xlint"
    ) ++ {
      // Add explicit compiler optimizations when building for Scala 2.13
      CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, 13)) => Seq("-opt:l:inline", "-opt-inline-from:<sources>")
        case _             => Seq.empty
      }
    },

    // 3. Spark & MLlib Dependency Matrix
    // Using %%% or explicit dynamic versioning to match Spark's release matrix
    libraryDependencies ++= {
      val sparkVersion = "3.5.1" // Target a stable Spark release that explicitly supports both

      Seq(
        "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
        "org.apache.spark" %% "spark-sql"  % sparkVersion % Provided,
        "org.apache.spark" %% "spark-mllib" % sparkVersion % Provided,

        // Logging
        "org.apache.logging.log4j" % "log4j-api"  % "2.25.4",
        "org.apache.logging.log4j" % "log4j-core" % "2.25.4",

        // Testing framework
        "org.scalatest" %% "scalatest" % "3.2.18" % Test,
        "com.holdenkarau" %% "spark-testing-base" % s"${sparkVersion}_1.5.3" % Test
      )
    },

    // 4. Spark Local Execution Test Settings
    Test / fork := true,
    Test / javaOptions ++= Seq(
      "-Xms1G",
      "-Xmx4G"
    )
  )