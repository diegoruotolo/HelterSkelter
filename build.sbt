import sbt._
import Keys._

lazy val root = (project in file("."))
  .settings(
    name := "helter-skelter",
    organization := "io.rolling.spark",
    version := "0.0.1-SNAPSHOT",

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
        "org.apache.logging.log4j" % "log4j-api" % "2.17.2",
        "org.apache.logging.log4j" % "log4j-core" % "2.17.2",

        // Testing framework
        "org.scalatest" %% "scalatest" % "3.2.18" % Test,
        "com.holdenkarau" %% "spark-testing-base" % s"${sparkVersion}_1.5.3" % Test
      )
    },

    // 4. Spark Local Execution Test Settings
    Test / fork := true,
    Test / javaOptions ++= Seq(
      "-Xms1G",
      "-Xmx4G",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED"
    )
  )